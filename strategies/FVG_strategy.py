# Standard Library Importe
from decimal import Decimal
from typing import Any
import sys
from pathlib import Path

# Nautilus Kern offizielle Importe (für Backtest eigentlich immer hinzufügen)
from nautilus_trader.trading import Strategy
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, TradeTick, QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.orders import MarketOrder, LimitOrder, StopMarketOrder
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderEvent, PositionEvent
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.identifiers import AccountId
from nautilus_trader.model.currencies import USDT, BTC
from nautilus_trader.common.enums import LogColor


from tools.help_funcs.base_strategy import BaseStrategy
from core.visualizing.backtest_visualizer_prototype import BacktestDataCollector
from tools.help_funcs.help_funcs_strategy import create_tags
from tools.structure.retest import RetestAnalyser
from tools.structure.fvg import FVG_Analyser
from tools.order_management.risk_manager import RiskManager
from tools.order_management.order_types import OrderTypes

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# Import new modular classes


class FVGStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size_usdt: Decimal
    risk_percent: float
    max_leverage: float
    min_account_balance: float
    risk_reward_ratio: float
    fvg_min_size: float
    fvg_lookback: int
    close_positions_on_stop: bool
    
class FVGStrategy(BaseStrategy, Strategy):
    def __init__(self, config: FVGStrategyConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.venue = self.instrument_id.venue
        self.trade_size_usdt = config.trade_size_usdt
        self.close_positions_on_stop = config.close_positions_on_stop
        self.venue = self.instrument_id.venue
        self.risk_manager = None
        self.bar_type = config.bar_type
        self.fvg_detector = FVG_Analyser(min_size=self.config.fvg_min_size, lookback=self.config.fvg_lookback)
        self.retest_analyser = RetestAnalyser()
        self.risk_manager = None  # Will be initialized with account balance
        self.bar_counter = 0

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        self.subscribe_bars(self.bar_type)
        self.subscribe_trade_ticks(self.instrument_id)
        self.subscribe_quote_ticks(self.instrument_id)
        self.log.info("Strategy started!")

        self.collector = BacktestDataCollector()
        self.collector.initialise_logging_indicator("position", 1)
        self.collector.initialise_logging_indicator("realized_pnl", 2)
        self.collector.initialise_logging_indicator("unrealized_pnl", 3)
        self.collector.initialise_logging_indicator("equity", 4)
        
        self.risk_manager = RiskManager(
            self,
            Decimal(str(self.config.risk_percent)),
            Decimal(str(self.config.max_leverage)),
            Decimal(str(self.config.min_account_balance)),
            Decimal(str(self.config.risk_reward_ratio))
        )
        self.order_types = OrderTypes(self)

    def get_position(self):
        return self.base_get_position()

    def on_bar(self, bar: Bar) -> None: 
        # Get account balance and update risk manager

        # Update FVG detector with new bar
        self.fvg_detector.update_bars(bar)
        
        # Check for new FVGs and set retest zones
        self.check_for_fvg()

        # Check for retest opportunities and execute trades
        self.check_for_bullish_retest(bar)
        self.check_for_bearish_retest(bar)

        # Update visualizer data
        self.update_visualizer_data(bar)

    def check_for_bullish_retest(self, bar: Bar) -> None:
        is_bullish_retest, bullish_zone = self.retest_analyser.check_box_retest_zone(price=bar.low, filter="long")
        
        if is_bullish_retest and self.risk_manager.check_if_balance_is_sufficient():
            self.log.info(f"Retest bullische FVG: {bullish_zone}")
            
            entry_price = bar.close
            stop_loss = bar.low
            
            position_size, is_position_valid = self.risk_manager.calculate_position_size(entry_price, stop_loss)

            if not is_position_valid:
                self.log.error("Invalid position size calculated. Skipping order submission.")
            else:
                self.execute_buy_order(entry_price, stop_loss, position_size)
                
            # Remove the retested zone
            self.retest_analyser.remove_box_retest_zone(bullish_zone["upper"], bullish_zone["lower"])

    def check_for_bearish_retest(self, bar: Bar) -> None:
        is_bearish_retest, bearish_zone = self.retest_analyser.check_box_retest_zone(price=bar.high, filter="short")
        
        if is_bearish_retest and self.risk_manager.check_if_balance_is_sufficient():
            self.log.info(f"Retest bearishe FVG: {bearish_zone}")
            
            entry_price = bar.close
            stop_loss = bar.high
            
            position_size, is_position_valid = self.risk_manager.calculate_position_size(entry_price, stop_loss)
            
            if not is_position_valid:
                self.log.error("Invalid position size calculated. Skipping order submission.")
            else:
                self.execute_sell_order(entry_price, stop_loss, position_size)

            # Remove the retested zone
            self.retest_analyser.remove_box_retest_zone(bearish_zone["upper"], bearish_zone["lower"])

    def execute_buy_order(self, entry_price: Decimal, stop_loss: Decimal, position_size: Decimal) -> None:
        position_size = round(position_size, self.instrument.size_precision)
        take_profit = self.risk_manager.calculate_tp_price(entry_price, stop_loss)
                    
        self.order_types.submit_long_bracket_order(position_size, entry_price, stop_loss, take_profit)
                    
        self.log.info(f"Order-Submit: Entry={entry_price}, SL={stop_loss}, TP={take_profit}, Size={position_size}")

    def execute_sell_order(self, entry_price: Decimal, stop_loss: Decimal, position_size: Decimal) -> None:
        position_size = round(position_size, self.instrument.size_precision)
        take_profit = self.risk_manager.calculate_tp_price(entry_price, stop_loss)
                    
        self.order_types.submit_short_bracket_order(position_size, entry_price, stop_loss, take_profit)
        
        self.log.info(f"Order-Submit: Entry={entry_price}, SL={stop_loss}, TP={take_profit}, Size={position_size}")

    def check_for_fvg(self):
        is_bullish_fvg, (fvg_high, fvg_low) = self.fvg_detector.is_bullish_fvg()
        if is_bullish_fvg:
            self.log.info(f"Bullische FVG erkannt: Gap von {fvg_high} bis {fvg_low}")
            self.retest_analyser.set_box_retest_zone(upper=fvg_low, lower=fvg_high, long_retest=True)

        # Check for new bearish FVG
        is_bearish_fvg, (fvg_high, fvg_low) = self.fvg_detector.is_bearish_fvg()
        if is_bearish_fvg:
            self.log.info(f"Bearishe FVG erkannt: Gap von {fvg_high} bis {fvg_low}") 
            self.retest_analyser.set_box_retest_zone(upper=fvg_high, lower=fvg_low, long_retest=False)

    def close_position(self) -> None:
        return self.base_close_position()
    
    def on_stop(self) -> None:
        return self.base_on_stop()

    def on_order_filled(self, order_filled) -> None:
        return self.base_on_order_filled(order_filled)

    def on_position_closed(self, position_closed) -> None:
        return self.base_on_position_closed(position_closed)

    def on_error(self, error: Exception) -> None:
        return self.base_on_error(error)

    def get_account_balance(self) -> Decimal:
        # Get account balance for risk manager
        account_id = AccountId("BINANCE-001")
        account = self.cache.account(account_id)
        usdt_free = account.balance(USDT).free
        if usdt_free is None:
            usdt_balance = Decimal("0")
        else:
            usdt_balance = Decimal(str(usdt_free).split(" ")[0])
        return usdt_balance

    def get_position(self):
        if hasattr(self, "cache") and self.cache is not None:
            positions = self.cache.positions_open(instrument_id=self.instrument_id)
            if positions:
                return positions[0]
        return None
    
    def update_visualizer_data(self, bar: Bar) -> None:
        net_position = self.portfolio.net_position(self.instrument_id)
        unrealized_pnl = self.portfolio.unrealized_pnl(self.instrument_id)
        self.realized_pnl = self.portfolio.realized_pnl(self.instrument_id)

        self.collector.add_indicator(timestamp=bar.ts_event, name="position", value=self.portfolio.net_position(self.instrument_id) if self.portfolio.net_position(self.instrument_id) is not None else None)
        self.collector.add_indicator(timestamp=bar.ts_event, name="unrealized_pnl", value=float(unrealized_pnl) if unrealized_pnl is not None else None)
        self.collector.add_indicator(timestamp=bar.ts_event, name="realized_pnl", value=float(self.realized_pnl) if self.realized_pnl is not None else None)
        self.collector.add_bar(timestamp=bar.ts_event, open_=bar.open, high=bar.high, low=bar.low, close=bar.close)
