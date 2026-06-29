from __future__ import annotations
import os
import csv
import numpy as np
from search.base import BaseSearch


class MiAlgoritmo(BaseSearch):
    def __init__(self, pop_size=100, n_gen=1000, problem=None, verbose=False, output_file=None):
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.verbose = verbose
        self.output_file = output_file
        self.n_eval = 1

    def _initialize_pop(self):
        """Genera la población inicial aleatoriamente"""
        x, x_f = [], []
        for _ in range(self.pop_size):
            ind = np.random.randint(0, 2, size=self.problem.n_var).tolist()
            x.append(ind)
            x_f.append(self.problem._evaluate_multi(ind, self.n_eval))
            self.n_eval += 1
        return {"X": x, "F": x_f}

    def _save_population(self, pop, generation: int):
        """Igual que en NSGA3, guarda el CSV"""
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

    def _non_dominated_samples(self, pop):
        """Retorna índices de soluciones no dominadas"""
        indexes = []
        for i, p in enumerate(pop["F"]):
            dominated = any(
                all(q_j <= p_j for q_j, p_j in zip(q, p)) and any(q_j < p_j for q_j, p_j in zip(q, p))
                for j, q in enumerate(pop["F"]) if i != j
            )
            if not dominated:
                indexes.append(i)
        return indexes

    def _do(self):
        # ─── Tu lógica de búsqueda va aquí ───
        pop = self._initialize_pop()
        self._save_population(pop, generation=0)

        for gen in range(1, self.n_gen + 1):
            # TODO: implementar tu algoritmo
            # Ejemplo: reemplazar el peor individuo por uno nuevo
            pass

            self.problem.notify_generation(gen)
            self._save_population(pop, generation=gen)

        nds_index = self._non_dominated_samples(pop)
        nds = {key: [pop[key][i] for i in nds_index] for key in pop}
        return pop, nds

    def run(self):
        return self._do()