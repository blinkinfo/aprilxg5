# Implementation Notes - Position Redeemer

## Key Architecture Facts
- Bot: BTC 5-min signal bot with XGBoost ML, Polymarket auto-trading
- Wallet: signature_type=2 (Gnosis Safe/Proxy), funder_address holds positions  
- web3>=6.14.0 already in requirements.txt
- POLYGON_RPC_URL needs to be added to PolymarketConfig
- polymarket_client.py uses: CLOB_HOST, GAMMA_API, DATA_API constants, httpx.AsyncClient
- polymarket_client._funder_address = the proxy wallet holding positions
- polymarket_client._private_key = EOA private key
- polymarket_client._http = httpx.AsyncClient(timeout=15)
- _parse_market() extracts: condition_id, neg_risk, clob_token_ids, outcomes, etc.

## Contract Addresses (Polygon Mainnet)
- CTF (Conditional Tokens): 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
- USDC.e: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174  
- Neg Risk Adapter: 0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296
- Neg Risk Wrapped Collateral: 0x3A3BD7bb9528E159577F7C2e685CC81A765002E2

## redeemPositions ABI
- Function: redeemPositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] indexSets)
- collateralToken: USDC.e for non-negRisk, NegRiskWrappedCollateral for negRisk
- parentCollectionId: bytes32(0) always
- conditionId: from market data (already in _parse_market)
- indexSets: [1, 2] redeems both outcomes (winner pays $1, loser pays $0)
- Burns ENTIRE token balance, no amount param

## Safe/Proxy Wallet Execution
- signature_type=2 means positions held by Safe proxy (funder_address)
- Must execute through Safe's execTransaction
- Safe execTransaction(address to, uint256 value, bytes data, uint8 operation, uint256 safeTxGas, uint256 baseGas, uint256 gasPrice, address gasToken, address payable refundReceiver, bytes signatures)
- For single-owner Safe: signatures = abi.encodePacked(uint256(0), uint256(owner_address), bytes32(0), uint8(1))
- Actually simpler: pre-approved signature format
- operation=0 (Call), all gas params = 0

## Data API for Redeemable Positions  
- GET https://data-api.polymarket.com/positions?user={funder_address}
- Response includes: conditionId, resolved, outcome, size fields
- Check market resolution via Gamma API: GET https://gamma-api.polymarket.com/markets?condition_id={id}

## Bot Integration Points
- Main loop: _main_loop() checks every 5s (config.main_loop_interval)
- Resolution: fires 30-90s into new candle
- Add redeemer on slower interval (every 120s)
- Callback pattern: set_callbacks() -> telegram_bot commands -> async callbacks in bot.py
- Format pattern: formatters.py has all HTML-formatted Telegram messages
- Config: PolymarketConfig dataclass, loaded in from_env()

## Telegram Bot Pattern
- _BOT_COMMANDS list for menu autocomplete
- CommandHandler registration in initialize()
- Callbacks set via set_callbacks()
- All messages HTML parse_mode
- send_message() handles splitting long messages

## Config.py PolymarketConfig
```python
@dataclass
class PolymarketConfig:
    private_key: str = ""          
    funder_address: str = ""       
    signature_type: int = 2        
    enabled: bool = False           
```
Need to add: polygon_rpc_url: str = "", auto_redeem: bool = True, redeem_check_interval: int = 120

## Files to Create/Modify
1. NEW: src/position_redeemer.py
2. MODIFY: src/config.py - add polygon_rpc_url, auto_redeem, redeem_check_interval to PolymarketConfig
3. MODIFY: src/formatters.py - add format_redemption(), format_redeem_error(), format_redeem_status()
4. MODIFY: src/bot.py - add redeemer init, _last_redeem_check_ts, redeem check in main loop, callbacks
5. MODIFY: src/telegram_bot.py - add /redeem command, callback
6. NO CHANGE: requirements.txt (web3 already there)
