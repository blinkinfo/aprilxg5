# Full Context for Position Redeemer Implementation

## config.py PolymarketConfig (line ~87-93)
```python
@dataclass
class PolymarketConfig:
    """Polymarket auto-trading configuration."""
    private_key: str = ""          # Wallet private key (hex)
    funder_address: str = ""       # Funder/proxy wallet address
    signature_type: int = 2        # Signature type (0, 1, or 2)
    enabled: bool = False           # Derived: True if private_key is set
```

## config.py from_env() polymarket section (line ~131-140)
```python
        config.polymarket.private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        config.polymarket.funder_address = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
        if os.environ.get("POLYMARKET_SIGNATURE_TYPE"):
            config.polymarket.signature_type = int(os.environ["POLYMARKET_SIGNATURE_TYPE"])
        config.polymarket.enabled = bool(config.polymarket.private_key)
```

## config.py BotConfig (line ~96-107)
```python
@dataclass
class BotConfig:
    mexc: MEXCConfig = field(default_factory=MEXCConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    prediction_lead_seconds: int = 15
    main_loop_interval: int = 5
    data_dir: str = "data"
    model_dir: str = "models"
    log_level: str = "INFO"
```

## bot.py key imports and class structure
```python
from .config import BotConfig
from .data_fetcher import MEXCFetcher
from .features import FeatureEngineer
from .model import PredictionModel
from .signal_tracker import SignalTracker
from .telegram_bot import TelegramBot
from . import formatters
from .polymarket_client import PolymarketClient
from .auto_trader import AutoTrader

class SignalBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.fetcher = MEXCFetcher(config.mexc)
        self.model = PredictionModel(config.model)
        self.tracker = SignalTracker(config.data_dir)
        self.telegram = TelegramBot(config.telegram)
        self._running = False
        self._last_signal_candle_ts = None
        self._last_resolved_candle_ts = None
        self.polymarket_client = None
        self.auto_trader = None
```

## bot.py start() - telegram callbacks (line ~62-73)
```python
        self.telegram.set_callbacks(
            stats_cb=self._get_stats_text,
            recent_cb=self._get_recent_text,
            status_cb=self._get_status_text,
            retrain_cb=self._retrain_model,
            autotrade_toggle_cb=self._toggle_autotrade,
            set_amount_cb=self._set_trade_amount,
            balance_cb=self._get_balance_text,
            positions_cb=self._get_positions_text,
            pmstatus_cb=self._get_pmstatus_text,
        )
```

## bot.py start() - polymarket init (line ~76-95)
```python
        if self.config.polymarket.enabled:
            self.polymarket_client = PolymarketClient(
                private_key=self.config.polymarket.private_key,
                funder_address=self.config.polymarket.funder_address,
                signature_type=self.config.polymarket.signature_type,
            )
            init_result = await self.polymarket_client.initialize()
            if init_result["success"]:
                self.auto_trader = AutoTrader(
                    polymarket_client=self.polymarket_client,
                    data_dir=self.config.data_dir,
                )
```

## bot.py _main_loop() structure (line ~112-140)
- while self._running loop
- gets now, current_slot, seconds_in_candle, seconds_until_close
- SIGNAL: 0 < seconds_until_close <= prediction_lead_seconds
- RESOLUTION: RESOLUTION_DELAY_SECONDS <= seconds_in_candle < RESOLUTION_WINDOW_END
- RETRAIN: self.model.needs_retrain()
- asyncio.sleep(self.config.main_loop_interval)

## bot.py stop() (line ~42-48)
```python
    async def stop(self):
        self._running = False
        await self.telegram.send_message(formatters.format_shutdown())
        await self.telegram.stop()
        if self.polymarket_client:
            await self.polymarket_client.close()
        await self.fetcher.close()
        self.model.save(self.config.model_dir)
```

## polymarket_client.py key constants and structure
```python
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CHAIN_ID = 137
SLOT_PERIOD = 300

class PolymarketClient:
    def __init__(self, private_key, funder_address, signature_type=2):
        self._private_key = private_key
        self._funder_address = funder_address
        self._signature_type = signature_type
        self._client = None  # ClobClient
        self._api_creds = None
        self._initialized = False
        self._last_traded_slot = None
        self._http = httpx.AsyncClient(timeout=15)
    
    # Methods: initialize(), get_balance(), get_market_for_slot(), 
    #          get_current_market(), get_best_price(), place_trade(),
    #          get_open_positions(), is_connected(), close()
    
    # _parse_market() extracts: condition_id, neg_risk, clob_token_ids, outcomes, etc.
    # get_open_positions() uses DATA_API /positions?user={funder_address}
```

## Contract Addresses (Polygon Mainnet)
- CTF (Conditional Tokens): 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
- USDC.e: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
- Neg Risk Adapter: 0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296
- Neg Risk Wrapped Collateral: 0x3A3BD7bb9528E159577F7C2e685CC81A765002E2

## Key Data API endpoint for redeemable positions
GET https://data-api.polymarket.com/positions?user={address}&redeemable=true
Returns positions with: conditionId, size, outcome, outcomeIndex, negativeRisk, redeemable=true, title, slug, asset

## CTF redeemPositions ABI
Function: redeemPositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] indexSets)
- collateralToken: USDC.e for non-negRisk, NegRiskWrappedCollateral for negRisk  
- parentCollectionId: bytes32(0) always
- conditionId: from market data
- indexSets: [1, 2] redeems both outcomes
- For negRisk markets: call NegRiskAdapter.redeemPositions() instead of CTF directly

## Safe/Proxy Wallet (signature_type=2)
- Positions held by funder_address (Safe proxy)
- Must call through Safe's execTransaction
- For single-owner Safe with pre-approved hash:
  signatures = 0x + 24 zeroes + owner_address(no 0x) + 64 zeroes + 01
- execTransaction(to, value=0, data, operation=0, safeTxGas=0, baseGas=0, gasPrice=0, gasToken=0x0, refundReceiver=0x0, signatures)

## Requirements.txt already has web3>=6.14.0

