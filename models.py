from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class TradeSignal:
    """
    Standardized signal object returned by strategies.
    """
    symbol: str
    timeframe: str
    strategy_name: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # 0.0 to 1.0
    price: float      # Price at time of signal
    sl: Optional[float] = None
    tp: Optional[float] = None
    features: Dict[str, Any] = field(default_factory=dict) # For ML
    metadata: Dict[str, Any] = field(default_factory=dict) # For debugging (e.g., "Swept low at X")
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        self.signal_type = self.signal_type.upper()
