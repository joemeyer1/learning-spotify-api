#!usr/bin/env python3


from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class ArtistInfo:
    name: str
    id: str


@dataclass
class TrackInfo:
    name: str
    id: str
    feats: Optional[Dict[str, float]] = None
    artist: Optional[str] = None

    def get_feats(self, target_feature_types: List[str]) -> List[float]:
        if self.feats is None:
            return []
        else:
            assert type(self.feats) == dict
            return [self.feats[feat_type] for feat_type in target_feature_types]