from __future__ import annotations
import numpy as np


class GenerationTracker:
    """
    Registra métricas del frente de Pareto al final de cada generación.
    Métricas:
      - Hipervolumen (HV)
      - IGD  (Inverted Generational Distance)
      - IGD+ (versión dominancia-aware)
      - Spread / Delta
      - Tamaño del frente de Pareto
    """

    def __init__(self, ref_point: list[float] | None = None):
        """
        ref_point : punto de referencia para HV.
                    Si es None se calcula dinámicamente (nadir + 10 %).
        """
        self.ref_point = np.array(ref_point) if ref_point is not None else None
        self.history: list[dict] = []   # una entrada por generación

    # Punto de referencia dinámico ------------------------------------------------------------------

    def _get_ref_point(self, pareto: np.ndarray) -> np.ndarray:
        if self.ref_point is not None:
            return self.ref_point
        nadir  = pareto.max(axis=0)
        margin = np.abs(nadir) * 0.1 + 1e-6
        return nadir + margin

    # Hipervolumen 2D exacto  ------------------------------------------------------------------

    def _hypervolume_2d(self, pareto: np.ndarray, ref: np.ndarray) -> float:
        pts = pareto[np.all(pareto < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        hv, prev_x = 0.0, ref[0]
        for pt in reversed(pts):
            hv    += (prev_x - pt[0]) * (ref[1] - pt[1])
            prev_x = pt[0]
        return hv

    # IGD e IGD+ ------------------------------------------------------------------

    def _igd(self, pareto: np.ndarray, reference_set: np.ndarray) -> float:
        """
        IGD: distancia media desde cada punto del reference_set
        al punto más cercano del frente aproximado (pareto).
        """
        if reference_set is None or len(reference_set) == 0:
            return 0.0
        dists = [
            np.min(np.linalg.norm(pareto - r, axis=1))
            for r in reference_set
        ]
        return float(np.mean(dists))

    def _igd_plus(self, pareto: np.ndarray, reference_set: np.ndarray) -> float:
        """
        IGD+: dominancia-aware.
        d+(z, a) = sqrt( sum( max(a_i - z_i, 0)^2 ) )
        """
        if reference_set is None or len(reference_set) == 0:
            return 0.0
        dists = []
        for r in reference_set:
            diff  = np.maximum(pareto - r, 0)          # solo penaliza si pareto es peor
            dists.append(np.min(np.linalg.norm(diff, axis=1)))
        return float(np.mean(dists))

    # Spread / Delta  ------------------------------------------------------------------

    def _spread(self, pareto: np.ndarray) -> float:
        """
        Spread de Deb: mide uniformidad de distribución del frente.
        Valor cercano a 0 → distribución uniforme.
        """
        if len(pareto) < 2:
            return 0.0

        pts = pareto[np.argsort(pareto[:, 0])]

        # Distancias entre puntos consecutivos
        diffs      = np.diff(pts, axis=0)
        distances  = np.linalg.norm(diffs, axis=1)
        d_mean     = distances.mean()

        # Distancias extremas (primer y último punto del frente ideal)
        d_first = np.linalg.norm(pts[0]  - pts[-1])  # proxy simple
        d_last  = d_first

        numerator   = d_first + d_last + np.sum(np.abs(distances - d_mean))
        denominator = d_first + d_last + len(distances) * d_mean

        return float(numerator / denominator) if denominator > 0 else 0.0
    # Spacing (Espaciamiento) ------------------------------------------------------------------

    def _spacing(self, pareto: np.ndarray) -> float:
        """
        Métrica de Espaciamiento (Spacing) de Schott.
        Mide la varianza de las distancias entre soluciones vecinas en el frente.
        Un valor cercano a 0 indica una distribución uniforme.
        """
        n = len(pareto)
        if n < 2:
            return 0.0

        dists = []
        for i in range(n):
            # Calculamos la distancia de Manhattan (L1) desde el punto 'i' a todos los demás
            diffs = np.sum(np.abs(pareto - pareto[i]), axis=1)
            
            # Reemplazamos la distancia a sí mismo (que es 0 en el índice 'i') por infinito 
            # para no seleccionarlo como el "vecino más cercano"
            diffs[i] = np.inf
            
            # Guardamos la distancia al vecino más cercano
            dists.append(np.min(diffs))
            
        dists = np.array(dists)
        
        # Si solo hay 2 puntos, la distancia entre ellos es la misma para ambos, por lo que la varianza es 0.
        if n == 2:
            return 0.0
            
        # Calculamos la desviación estándar muestral (ddof=1 equivale a dividir entre n-1)
        return float(np.std(dists, ddof=1))

    # Filtro de no-dominancia (para obtener el frente de la población) ------------------------------------------------------------------

    def _pareto_front(self, objectives: list) -> np.ndarray:
        objs = np.array(objectives, dtype=float)
        n    = len(objs)
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_nd[i]:
                continue
            dominated = (
                np.all(objs <= objs[i], axis=1) &
                np.any(objs <  objs[i], axis=1)
            )
            dominated[i] = False
            if dominated.any():
                is_nd[i] = False
        return objs[is_nd]

    # API principal: llamar al final de cada generación ------------------------------------------------------------------

    def record(
        self,
        generation: int,
        objectives: list,
        reference_set: np.ndarray | None = None,
    ) -> None:
        """
        Calcula y almacena todas las métricas para la generación dada.

        generation    : número de generación actual
        objectives    : lista de objetivos de TODA la población actual
        reference_set : frente de Pareto verdadero (opcional, para IGD/IGD+)
        """
        pareto = self._pareto_front(objectives)
        ref    = self._get_ref_point(pareto)

        hv     = self._hypervolume_2d(pareto, ref)
        #igd    = self._igd(pareto, reference_set)
        #igd_p  = self._igd_plus(pareto, reference_set)
        spread = self._spread(pareto)
        spacing = self._spacing(pareto)
        pf_size = len(pareto)

        self.history.append({
            "generation":  generation,
            "hv":          hv,
            #"igd":         igd,
            #"igd_plus":    igd_p,
            "spread":      spread,
            "spacing":     spacing,
            "pareto_size": pf_size,
        })

    # Exportar a CSV ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        import csv
        if not self.history:
            return
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
        print(f"[INFO] Metrics log saved to: {filepath}")