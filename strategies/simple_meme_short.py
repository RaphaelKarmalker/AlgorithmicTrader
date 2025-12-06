from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Union
from nautilus_trader.trading import Strategy
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.common.enums import LogColor
from pydantic import Field
from decimal import Decimal
from nautilus_trader.indicators.volatility import AverageTrueRange
from tools.help_funcs.base_strategy import BaseStrategy
from tools.order_management.order_types import OrderTypes
from tools.order_management.risk_manager import RiskManager

class SimpleMemeShortConfig(StrategyConfig):
    instruments: List[dict]
    min_account_balance: float
    run_id: str
    
    # Risk Params
    sl_atr_multiple: float = 2.0
    atr_period: int = 14
    
    # Baseline Entry Params
    entry_threshold_pct: float = 0.005 # 0.5% below VWAP
    min_bars_history: int = 10 # Wait 2.5 hours (10 * 15m)
    
    # Exit Params
    trailing_stop_atr: float = 3.0
    max_hold_bars: int = 100

    # Standard Boilerplate
    log_growth_atr_risk: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "atr_period": 30,
            "atr_multiple": 2.5,
            "risk_percent": 0.01
        }
    )
    exp_growth_atr_risk: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    exp_fixed_trade_risk: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    log_fixed_trade_risk: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    
    only_execute_short: bool = True
    hold_profit_for_remaining_days: bool = False
    close_positions_on_stop: bool = True
    max_concurrent_positions: int = 50
    max_leverage: Decimal = 10.0

class SimpleMemeShort(BaseStrategy, Strategy):
    """
    A baseline strategy for shorting meme coins post-listing.
    Logic:
    1. Calculate Anchored VWAP (from start of data).
    2. If Price < VWAP * (1 - threshold): SHORT.
    3. No complex filters. If it's below the average price, we short.
    """
    def __init__(self, config: SimpleMemeShortConfig):
        super().__init__(config)
        self.risk_manager = RiskManager(config)
        self.risk_manager.set_strategy(self)    
        self.risk_manager.set_max_leverage(Decimal(str(config.max_leverage)))
        self.order_types = OrderTypes(self) 
        self.add_instrument_context()

    def add_instrument_context(self):
        for current_instrument in self.instrument_dict.values():
            # Risk Indicators
            atr_period = self.config.atr_period
            current_instrument["atr"] = AverageTrueRange(atr_period)
            
            # VWAP State
            current_instrument["cum_pv"] = 0.0
            current_instrument["cum_vol"] = 0.0
            current_instrument["vwap"] = None
            current_instrument["bars_seen"] = 0

            # Position State
            current_instrument["sl_price"] = None
            current_instrument["trailing_stop_price"] = None
            current_instrument["best_price"] = None
            current_instrument["bars_held"] = 0

            # Logging
            # current_instrument["collector"].initialise_logging_indicator("vwap", 1)

    def on_start(self):
        super().on_start()
        # Request minimal history just to warm up ATR
        bars_needed = 50
        for instrument_data in self.config.instruments:
            try:
                instrument_id = InstrumentId.from_str(instrument_data.get("instrument_id"))
                bar_type = BarType.from_str(instrument_data.get("bar_types")[0])
                start_time = self._clock.utc_now() - timedelta(minutes=bars_needed * 15)
                self.request_bars(bar_type, start=start_time)
            except Exception:
                pass

    def on_bar(self, bar: Bar) -> None:
        instrument_id = bar.bar_type.instrument_id
        current_instrument = self.instrument_dict.get(instrument_id)
        if not current_instrument: return
        if "atr" not in current_instrument: self.add_instrument_context()

        # 1. Update Indicators
        self.update_indicators(bar, current_instrument)
        self.base_collect_bar_data(bar, current_instrument)

        # 2. Check Position
        position = self.base_get_position(instrument_id)
        if position and position.side == PositionSide.SHORT:
            self.manage_short_position(bar, current_instrument, position)
        elif not position:
            self.check_entry(bar, current_instrument)

    def update_indicators(self, bar: Bar, ctx: Dict[str, Any]):
        # ATR
        ctx["atr"].handle_bar(bar)
        
        # Anchored VWAP
        price = float(bar.close)
        volume = float(bar.volume)
        typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        
        ctx["cum_pv"] += typical * volume
        ctx["cum_vol"] += volume
        ctx["bars_seen"] += 1
        
        if ctx["cum_vol"] > 0:
            vwap = ctx["cum_pv"] / ctx["cum_vol"]
            ctx["vwap"] = vwap
            # ctx["collector"].add_indicator(bar.ts_event, "vwap", vwap)

    def check_entry(self, bar: Bar, ctx: Dict[str, Any]):
        # Wait for minimal history
        if ctx["bars_seen"] < self.config.min_bars_history:
            return

        vwap = ctx["vwap"]
        if vwap is None: return

        price = float(bar.close)
        threshold = self.config.entry_threshold_pct
        
        # BASELINE LOGIC: Price is below VWAP -> Short.
        if price < vwap * (1 - threshold):
            self.execute_short(bar, ctx)

    def execute_short(self, bar: Bar, ctx: Dict[str, Any]):
        if not self.can_open_new_position(): return
        
        instrument_id = bar.bar_type.instrument_id
        price = float(bar.close)
        atr = float(ctx["atr"].value) if ctx["atr"].initialized else price * 0.05
        
        # Set Initial Stop Loss
        sl_dist = atr * self.config.sl_atr_multiple
        sl_price = price + sl_dist
        
        qty = self.calculate_risk_based_position_size(instrument_id, price, sl_price)
        if qty > 0:
            self.order_types.submit_short_market_order_with_sl(instrument_id, qty, sl_price)
            ctx["sl_price"] = sl_price
            ctx["best_price"] = price
            ctx["trailing_stop_price"] = None
            ctx["bars_held"] = 0
            self.log.info(f"SHORT ENTRY: {instrument_id} @ {price} (VWAP: {ctx['vwap']:.4f})", LogColor.RED)

    def manage_short_position(self, bar: Bar, ctx: Dict[str, Any], position):
        ctx["bars_held"] += 1
        price = float(bar.close)
        
        # 1. Check Hard SL (Safety)
        if ctx["sl_price"] and float(bar.high) >= ctx["sl_price"]:
            # OrderTypes handles the actual SL order, we just reset state if we detect it might have hit
            # But to be safe, we can force close if we are still here
            pass 

        # 2. Trailing Stop Logic
        if ctx["best_price"] is None or price < ctx["best_price"]:
            ctx["best_price"] = price
        
        atr = float(ctx["atr"].value) if ctx["atr"].initialized else price * 0.05
        trail_dist = atr * self.config.trailing_stop_atr
        new_trail = ctx["best_price"] + trail_dist
        
        # Tighten trail downwards
        if ctx["trailing_stop_price"] is None or new_trail < ctx["trailing_stop_price"]:
            ctx["trailing_stop_price"] = new_trail
            
        # Check Trail Hit
        if ctx["trailing_stop_price"] and price >= ctx["trailing_stop_price"]:
            self.close_position(bar, position, "TRAILING_STOP")
            return

        # 3. Time Exit
        if ctx["bars_held"] > self.config.max_hold_bars:
            self.close_position(bar, position, "TIME_LIMIT")

    def close_position(self, bar: Bar, position, reason: str):
        instrument_id = bar.bar_type.instrument_id
        qty = abs(float(position.quantity))
        self.log.info(f"EXIT {reason}: {instrument_id} closing {qty}", LogColor.GREEN)
        self.order_types.submit_long_market_order(instrument_id, int(qty))
        
        # Cancel any open SL orders
        for order in self.cache.orders_open(instrument_id=instrument_id):
            self.cancel_order(order)

    def can_open_new_position(self) -> bool:
        open_positions = len([p for p in self.cache.positions() if p.is_open])
        return open_positions < self.config.max_concurrent_positions
