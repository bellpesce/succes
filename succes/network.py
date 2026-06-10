"""
succes/network.py  — Transmission network with PTDF support
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class TransmissionLink:
    """
    Directional transmission link between two regions.

    reactance = 0 → HVDC link: uses bilateral ATC only, no loop flows.
    reactance > 0 → AC link:   participates in PTDF computation.
    """
    region_a:    str
    region_b:    str
    max_mw_ab:   float
    max_mw_ba:   float  = -1.0
    loss_factor: float  = 0.0
    reactance:   float  = 0.0   # per-unit on 100 MVA base; 0 = HVDC
    is_hvdc:     bool   = False

    def __post_init__(self):
        if self.max_mw_ba < 0:
            self.max_mw_ba = self.max_mw_ab
        if self.reactance == 0.0:
            self.is_hvdc = True

    def capacity(self, from_region: str, to_region: str) -> float:
        if from_region == self.region_a and to_region == self.region_b:
            return self.max_mw_ab
        if from_region == self.region_b and to_region == self.region_a:
            return self.max_mw_ba
        raise ValueError(
            f"Link {self.region_a}<->{self.region_b} does not connect "
            f"{from_region} to {to_region}"
        )

    def involves(self, region: str) -> bool:
        return region in (self.region_a, self.region_b)

    def other(self, region: str) -> str:
        if region == self.region_a:
            return self.region_b
        if region == self.region_b:
            return self.region_a
        raise ValueError(f"{region} not on this link")


class Network:
    """
    Collection of TransmissionLinks.

    Call build_ptdf(regions, slack) after all links are added to
    precompute the zonal PTDF matrix for AC lines.
    HVDC links always use bilateral ATC regardless.
    """

    def __init__(self):
        self._links: list[TransmissionLink] = []
        self._ptdf_matrix: Optional[np.ndarray] = None
        self._ptdf_regions: Optional[list[str]] = None
        self._ptdf_ram: Optional[np.ndarray] = None

    def add_link(self, link: TransmissionLink) -> None:
        self._links.append(link)
        self._ptdf_matrix = None   # invalidate cache

    def links(self) -> list[TransmissionLink]:
        return list(self._links)

    def links_for_region(self, region: str) -> list[TransmissionLink]:
        return [l for l in self._links if l.involves(region)]

    def neighbours(self, region: str) -> list[str]:
        return [l.other(region) for l in self.links_for_region(region)]

    def capacity(self, from_region: str, to_region: str) -> float:
        for l in self._links:
            if l.involves(from_region) and l.involves(to_region):
                try:
                    return l.capacity(from_region, to_region)
                except ValueError:
                    continue
        return 0.0

    def loss_factor(self, from_region: str, to_region: str) -> float:
        for l in self._links:
            if l.involves(from_region) and l.involves(to_region):
                return l.loss_factor
        return 0.0

    def is_connected(self) -> bool:
        return len(self._links) > 0

    # ── PTDF ──────────────────────────────────────────────────────────────────

    def build_ptdf(
        self,
        regions: list[str],
        slack_region: str,
        ram_fraction: float = 0.80,
    ) -> None:
        """
        Compute zonal PTDF matrix using DC load-flow approximation.

        PTDF[l, r] = fraction of a 1 MW net injection at zone r
                     (withdrawn at slack) that flows on AC line l.

        Derivation:
          B_bus[i,i] = sum(1/X) for all AC lines at node i
          B_bus[i,j] = -1/X for AC line between i and j
          B_line[l, i] = +1/X_l,  B_line[l, j] = -1/X_l  for line l=(i,j)
          Remove slack row/col:
          PTDF = B_line_reduced @ inv(B_bus_reduced)
          Insert zero column for slack.

        Parameters
        ----------
        regions      : list of all regions (must cover all AC nodes)
        slack_region : reference bus; its PTDF column = 0 by convention
        ram_fraction : RAM = ram_fraction × min(atc_ab, atc_ba)
        """
        R = len(regions)
        ridx = {r: i for i, r in enumerate(regions)}
        slack_idx = ridx[slack_region]

        ac_links = [l for l in self._links if not l.is_hvdc]
        L_ac = len(ac_links)

        if L_ac == 0:
            self._ptdf_matrix  = np.zeros((0, R))
            self._ptdf_regions = list(regions)
            self._ptdf_ram     = np.zeros(0)
            return

        # Build B_bus (R × R) and B_line (L_ac × R)
        B_bus  = np.zeros((R, R))
        B_line = np.zeros((L_ac, R))
        for idx, lk in enumerate(ac_links):
            ia = ridx[lk.region_a]
            ib = ridx[lk.region_b]
            b  = 1.0 / lk.reactance
            B_bus[ia, ia] += b;  B_bus[ib, ib] += b
            B_bus[ia, ib] -= b;  B_bus[ib, ia] -= b
            B_line[idx, ia] =  b
            B_line[idx, ib] = -b

        # Remove slack row/col
        non_slack  = [i for i in range(R) if i != slack_idx]
        B_reduced  = B_bus[np.ix_(non_slack, non_slack)]
        B_line_r   = B_line[:, non_slack]

        try:
            B_inv = np.linalg.inv(B_reduced)
        except np.linalg.LinAlgError:
            B_inv = np.linalg.pinv(B_reduced)

        PTDF_r = B_line_r @ B_inv          # (L_ac × R-1)
        # Insert zero column at slack position
        PTDF = np.insert(PTDF_r, slack_idx, 0.0, axis=1)   # (L_ac × R)

        # RAM per AC line
        ram = np.array([
            ram_fraction * min(lk.max_mw_ab, lk.max_mw_ba)
            for lk in ac_links
        ])

        self._ptdf_matrix  = PTDF
        self._ptdf_regions = list(regions)
        self._ptdf_ram     = ram

        n_hvdc = len(self._links) - L_ac
        print(f"  [PTDF] Built {L_ac}×{R} matrix "
              f"({L_ac} AC lines, {n_hvdc} HVDC bilateral ATC)")

    @property
    def has_ptdf(self) -> bool:
        return (self._ptdf_matrix is not None
                and self._ptdf_matrix.shape[0] > 0)

    def ptdf_payload(self, regions: list[str]) -> dict:
        """
        Return PTDF data formatted for the Julia payload.
        regions: the ordered list of regions the Julia engine uses.
        Returns {} if PTDF not built or no AC lines.
        """
        if not self.has_ptdf:
            return {}

        # Reorder columns to match Julia's region order
        src_order = {r: i for i, r in enumerate(self._ptdf_regions)}
        col_map   = [src_order[r] for r in regions]
        PTDF_ordered = self._ptdf_matrix[:, col_map]

        return {
            "ptdf_matrix": PTDF_ordered.tolist(),  # (L_ac × R) row-major
            "ptdf_ram":    self._ptdf_ram.tolist(), # (L_ac,)
        }

    def __repr__(self) -> str:
        ac  = sum(1 for l in self._links if not l.is_hvdc)
        hv  = sum(1 for l in self._links if l.is_hvdc)
        ptdf = " [PTDF built]" if self.has_ptdf else ""
        return f"Network({len(self._links)} links: {ac} AC, {hv} HVDC){ptdf}"


# ── Legacy helper ─────────────────────────────────────────────────────────────

def compute_flows(
    surplus: np.ndarray,
    region_names: list[str],
    network: Network,
    congestion_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    S = surplus.shape[0]
    adj      = surplus.copy()
    con_cost = np.zeros(S)
    if not network.is_connected():
        return adj, con_cost
    region_idx = {r: i for i, r in enumerate(region_names)}
    for s in range(S):
        for link in network.links():
            ia = region_idx[link.region_a]
            ib = region_idx[link.region_b]
            sa, sb = adj[s, ia], adj[s, ib]
            if sa > 0 and sb < 0:
                flow = min(sa, -sb, link.max_mw_ab)
                adj[s, ia] -= flow
                adj[s, ib] += flow * (1.0 - link.loss_factor)
            elif sb > 0 and sa < 0:
                flow = min(sb, -sa, link.max_mw_ba)
                adj[s, ib] -= flow
                adj[s, ia] += flow * (1.0 - link.loss_factor)
    return adj, con_cost
