from __future__ import annotations
import os
import csv
import numpy as np
from tqdm import tqdm
from search.base import BaseSearch


class CMOPSO(BaseSearch):
    """
    Comprehensive Multi-Objective Particle Swarm Optimization (CMOPSO)
    adaptado para espacios de búsqueda binarios (NAS).

    Cada partícula tiene:
      - posición binaria X  (arquitectura)
      - velocidad continua V (probabilidad de flip por bit)
      - mejor personal pBest
    El repositorio externo mantiene las soluciones no dominadas.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 1000,
        verbose: bool = False,
        output_file: str | None = None,
        repo_size: int = 100,      # tamaño máximo del repositorio externo
        w: float = 0.9,            # inercia
        c1: float = 2.0,           # componente cognitivo (pBest)
        c2: float = 2.0,           # componente social (gBest del repositorio)
        v_max: float = 4.0,        # clamp de velocidad (logit scale)
    ):
        super().__init__(problem, pop_size, n_gen, verbose, output_file)
        self.repo_size = repo_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.n_eval = 1

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _initialize(self):
        n = self.problem.n_var

        positions = np.random.randint(0, 2, size=(self.pop_size, n)).astype(float)
        velocities = np.random.uniform(-1.0, 1.0, size=(self.pop_size, n))

        objectives = []
        for i in range(self.pop_size):
            f = self.problem._evaluate_multi(positions[i].astype(int).tolist(), self.n_eval)
            objectives.append(f)
            self.n_eval += 1

        objectives = np.array(objectives, dtype=float)

        # pBest inicializado igual a la posición inicial
        p_best_pos = positions.copy()
        p_best_obj = objectives.copy()

        return positions, velocities, objectives, p_best_pos, p_best_obj

    # ------------------------------------------------------------------
    # Repositorio externo (archivo de Pareto)
    # ------------------------------------------------------------------

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True si a domina a b (minimización)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _update_repository(self, repo_pos, repo_obj, new_pos, new_obj):
        """Agrega nuevas soluciones no dominadas al repositorio y limpia dominadas."""
        for pos, obj in zip(new_pos, new_obj):
            # ¿La nueva solución es dominada por alguna del repositorio?
            if any(self._dominates(r_obj, obj) for r_obj in repo_obj):
                continue
            # Elimina las del repositorio dominadas por la nueva
            keep = [
                i for i, r_obj in enumerate(repo_obj)
                if not self._dominates(obj, r_obj)
            ]
            repo_pos = [repo_pos[i] for i in keep]
            repo_obj = [repo_obj[i] for i in keep]
            repo_pos.append(pos.copy())
            repo_obj.append(obj.copy())

        # Si el repositorio supera su tamaño máximo → poda por crowding distance
        if len(repo_obj) > self.repo_size:
            repo_pos, repo_obj = self._prune_repository(repo_pos, repo_obj)

        return repo_pos, repo_obj

    def _crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        n, m = objectives.shape
        dist = np.zeros(n)
        for obj_idx in range(m):
            order = np.argsort(objectives[:, obj_idx])
            dist[order[0]] = dist[order[-1]] = np.inf
            obj_range = objectives[order[-1], obj_idx] - objectives[order[0], obj_idx]
            if obj_range == 0:
                continue
            for k in range(1, n - 1):
                dist[order[k]] += (
                    objectives[order[k + 1], obj_idx] - objectives[order[k - 1], obj_idx]
                ) / obj_range
        return dist

    def _prune_repository(self, repo_pos, repo_obj):
        """Elimina la solución con menor crowding distance hasta llegar a repo_size."""
        while len(repo_obj) > self.repo_size:
            objs = np.array(repo_obj)
            cd = self._crowding_distance(objs)
            worst = int(np.argmin(cd))
            repo_pos.pop(worst)
            repo_obj.pop(worst)
        return repo_pos, repo_obj

    # ------------------------------------------------------------------
    # Selección de guía (líder) del repositorio
    # ------------------------------------------------------------------

    def _select_leader(self, repo_pos, repo_obj) -> np.ndarray:
        """
        Torneo binario por crowding distance:
        selecciona al líder más aislado (mayor CD) para diversidad.
        """
        if len(repo_pos) == 1:
            return repo_pos[0].copy()
        objs = np.array(repo_obj)
        cd = self._crowding_distance(objs)
        i, j = np.random.choice(len(repo_pos), size=2, replace=False)
        winner = i if cd[i] >= cd[j] else j
        return repo_pos[winner].copy()

    # ------------------------------------------------------------------
    # Actualización de velocidad y posición (binario via sigmoid)
    # ------------------------------------------------------------------

    def _sigmoid(self, v: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))

    def _update_particle(self, pos, vel, p_best, leader):
        r1 = np.random.rand(self.problem.n_var)
        r2 = np.random.rand(self.problem.n_var)

        new_vel = (
            self.w * vel
            + self.c1 * r1 * (p_best - pos)
            + self.c2 * r2 * (leader  - pos)
        )
        # Clamp velocidad
        new_vel = np.clip(new_vel, -self.v_max, self.v_max)

        # Posición binaria via sigmoid
        prob = self._sigmoid(new_vel)
        new_pos = (np.random.rand(self.problem.n_var) < prob).astype(float)

        return new_pos, new_vel

    # Mutación de diversidad
    # ------------------------------------------------------------------

    def _mutate(self, pos: np.ndarray, gen: int) -> np.ndarray:
        """Bit-flip con probabilidad decreciente según la generación."""
        mut_rate = 0.01 / self.problem.n_var * (1 - gen / self.n_gen)
        mask = np.random.rand(self.problem.n_var) < mut_rate
        mutated = pos.copy()
        mutated[mask] = 1.0 - mutated[mask]
        return mutated

    # ------------------------------------------------------------------
    # Actualización de pBest
    # ------------------------------------------------------------------

    def _update_pbest(self, pos, obj, p_best_pos, p_best_obj):
        """Actualiza pBest usando dominancia de Pareto."""
        if self._dominates(obj, p_best_obj):
            return pos.copy(), obj.copy()
        if self._dominates(p_best_obj, obj):
            return p_best_pos, p_best_obj
        # No dominancia mutua → mantiene pBest actual (conservador)
        return p_best_pos, p_best_obj

    # ------------------------------------------------------------------
    # Guardado CSV (igual que NSGA3)
    # ------------------------------------------------------------------

    def _save_population(self, pop: dict, generation: int):
        if not self.output_file:
            return
        file_exists = os.path.exists(self.output_file)
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "generation", "individual_id",
                    "decoded_architecture", "predicted_psnr", "params",
                ])
            for idx, (x, obj) in enumerate(zip(pop["X"], pop["F"])):
                decoded_arch = self.problem.get_decoded_ind(x)
                writer.writerow([
                    generation, idx,
                    " ".join(map(str, decoded_arch)),
                    float(obj[0]), int(obj[1]),
                ])

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def _do(self):
        positions, velocities, objectives, p_best_pos, p_best_obj = self._initialize()

        # Inicializar repositorio con la población inicial
        repo_pos, repo_obj = self._update_repository(
            [], [], list(positions), list(objectives)
        )

        self._save_population(
            {"X": [p.astype(int).tolist() for p in positions], "F": list(objectives)},
            generation=0,
        )

        pbar = tqdm(total=self.n_gen, desc="CMOPSO Progress", unit="gen") if self.verbose else None

        for gen in range(1, self.n_gen + 1):
            for i in range(self.pop_size):
                leader = self._select_leader(repo_pos, repo_obj)

                new_pos, new_vel = self._update_particle(
                    positions[i], velocities[i], p_best_pos[i], leader
                )
                new_pos = self._mutate(new_pos, gen)

                new_obj = np.array(
                    self.problem._evaluate_multi(new_pos.astype(int).tolist(), self.n_eval)
                )
                self.n_eval += 1

                # Actualizar pBest
                p_best_pos[i], p_best_obj[i] = self._update_pbest(
                    new_pos, new_obj, p_best_pos[i], p_best_obj[i]
                )

                positions[i] = new_pos
                velocities[i] = new_vel
                objectives[i] = new_obj

            # Actualizar repositorio con la nueva generación
            repo_pos, repo_obj = self._update_repository(
                repo_pos, repo_obj, list(positions), list(objectives)
            )
            self.problem.notify_generation(gen)
            self._save_population(
                {"X": [p.astype(int).tolist() for p in positions], "F": list(objectives)},
                generation=gen,
            )

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # Población final + soluciones no dominadas (el repositorio ES el frente de Pareto)
        final_pop = {
            "X": [p.astype(int).tolist() for p in positions],
            "F": list(objectives),
        }
        nds = {
            "X": [p.astype(int).tolist() for p in repo_pos],
            "F": [o.tolist() for o in repo_obj],
        }
        return final_pop, nds

    def run(self):
        return self._do()