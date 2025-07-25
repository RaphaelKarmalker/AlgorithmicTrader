# Standard Library Importe
from decimal import Decimal
import time
from typing import Any
 
# Nautilus Kern Importe (für Backtest eigentlich immer hinzufügen)
from nautilus_trader.trading import Strategy
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.model.data import Bar, TradeTick, BarType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity, Currency
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.common.enums import LogColor

# Nautilus Strategie spezifische Importe
from tools.help_funcs.base_strategy import BaseStrategy
from tools.structure.TTTbreakout import TTTBreakout_Analyser
from tools.order_management.order_types import OrderTypes
from tools.order_management.risk_manager import RiskManager
from core.visualizing.backtest_visualizer_prototype import BacktestDataCollector
from tools.help_funcs.help_funcs_strategy import create_tags
from nautilus_trader.common.enums import LogColor

# Strategiespezifische Importe
from nautilus_trader.indicators.rsi import RelativeStrengthIndex

# ab hier der Code für die Strategie
class RSISimpleStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size_usdt: Decimal
    risk_percent: float
    max_leverage: float
    min_account_balance: float
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    close_positions_on_stop: bool = True
    
     
class RSISimpleStrategy(BaseStrategy, Strategy):
    def __init__(self, config: RSISimpleStrategyConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.trade_size_usdt = config.trade_size_usdt
        self.close_positions_on_stop = config.close_positions_on_stop
        self.venue = self.instrument_id.venue
        self.risk_manager = None
        self.bar_type = config.bar_type
        self.rsi_period = config.rsi_period
        self.rsi_overbought = config.rsi_overbought
        self.rsi_oversold = config.rsi_oversold
        self.rsi = RelativeStrengthIndex(period=self.rsi_period)
        self.last_rsi_cross = None
        self.stopped = False 
        self.realized_pnl = 0 

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        self.subscribe_bars(self.bar_type)
        #self.subscribe_trade_ticks(self.instrument_id)
        #self.subscribe_quote_ticks(self.instrument_id)
        self.log.info("Strategy started!")
        self.collector = BacktestDataCollector()
        self.collector.initialise_logging_indicator("RSI", 1)
        self.collector.initialise_logging_indicator("position", 2)
        self.collector.initialise_logging_indicator("realized_pnl", 3)
        self.collector.initialise_logging_indicator("unrealized_pnl", 4)
        self.collector.initialise_logging_indicator("equity", 5)

        self.risk_manager = RiskManager(
            self,
            Decimal(str(self.config.risk_percent)),
            Decimal(str(self.config.max_leverage)),
            Decimal(str(self.config.min_account_balance)),
        )
        self.order_types = OrderTypes(self)

        
    def get_position(self):
        return self.base_get_position()

    def on_bar(self, bar: Bar) -> None:
        self.rsi.handle_bar(bar)
        if not self.rsi.initialized:
            return

        open_orders = self.cache.orders_open(instrument_id=self.instrument_id)
        if open_orders:
            return 
        
        self.entry_logic(bar)
        self.collector.add_bar(timestamp=bar.ts_event, open_=bar.open, high=bar.high, low=bar.low, close=bar.close)
        self.update_visualizer_data(bar)

    def entry_logic(self, bar: Bar):
        trade_size_usdt = float(self.config.trade_size_usdt)
        qty = max(1, int(float(trade_size_usdt) // float(bar.close)))
        rsi_value = self.rsi.value
        position = self.get_position()
        if rsi_value > self.rsi_overbought:
            if self.last_rsi_cross is not "rsi_overbought":
                self.close_position()
                self.order_types.submit_short_market_order(qty)
            self.last_rsi_cross = "rsi_overbought"
        if rsi_value < self.rsi_oversold:
            if self.last_rsi_cross is not "rsi_oversold":
                self.close_position()
                self.order_types.submit_long_market_order(qty)
            self.last_rsi_cross = "rsi_oversold"

    def update_visualizer_data(self, bar: Bar) -> None:
        net_position = self.portfolio.net_position(self.instrument_id)
        unrealized_pnl = self.portfolio.unrealized_pnl(self.instrument_id)
        venue = self.instrument_id.venue
        account = self.portfolio.account(venue)
        usdt_balance = account.balance_total()
        equity = usdt_balance.as_double() + (float(unrealized_pnl) if unrealized_pnl else 0)

        rsi_value = float(self.rsi.value) if self.rsi.value is not None else None

        self.collector.add_indicator(timestamp=bar.ts_event, name="RSI", value=rsi_value)
        self.collector.add_indicator(timestamp=bar.ts_event, name="position", value=net_position)
        self.collector.add_indicator(timestamp=bar.ts_event, name="unrealized_pnl", value=float(unrealized_pnl) if unrealized_pnl else None)
        self.collector.add_indicator(timestamp=bar.ts_event, name="realized_pnl", value=float(self.realized_pnl) if self.realized_pnl else None)
        self.collector.add_indicator(timestamp=bar.ts_event, name="equity", value=equity)


    def close_position(self) -> None:
        return self.base_close_position()
    
    def on_stop(self) -> None:
        self.base_on_stop()

        self.stopped = True  
        net_position = self.portfolio.net_position(self.instrument_id)
        unrealized_pnl = self.portfolio.unrealized_pnl(self.instrument_id)  # Unrealized PnL
        realized_pnl = float(self.portfolio.realized_pnl(self.instrument_id))  # Unrealized PnL
        self.realized_pnl += unrealized_pnl+realized_pnl if unrealized_pnl is not None else 0
        unrealized_pnl = 0
        venue = self.instrument_id.venue
        account = self.portfolio.account(venue)
        usdt_balance = account.balance_total()
        equity = usdt_balance.as_double() + (float(unrealized_pnl) if unrealized_pnl else 0)

        self.collector.add_indicator(timestamp=self.clock.timestamp_ns(), name="equity", value=equity)
        self.collector.add_indicator(timestamp=self.clock.timestamp_ns(), name="position", value=self.portfolio.net_position(self.instrument_id) if self.portfolio.net_position(self.instrument_id) is not None else None)
        self.collector.add_indicator(timestamp=self.clock.timestamp_ns(), name="unrealized_pnl", value=float(unrealized_pnl) if unrealized_pnl is not None else None)
        self.collector.add_indicator(timestamp=self.clock.timestamp_ns(), name="realized_pnl", value=float(self.realized_pnl) if self.realized_pnl is not None else None)
        logging_message = self.collector.save_data()
        self.log.info(logging_message, color=LogColor.GREEN)

        #self.collector.visualize()  # Visualize the data if enabled
    def on_order_filled(self, order_filled) -> None:
        return self.base_on_order_filled(order_filled)

    def on_position_closed(self, position_closed) -> None:
        return self.base_on_position_closed(position_closed)

    def on_error(self, error: Exception) -> None:
        return self.base_on_error(error)

