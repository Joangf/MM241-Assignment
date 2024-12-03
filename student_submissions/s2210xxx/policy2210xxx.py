from policy import Policy
import numpy as np
from scipy.optimize import linprog


class LP(Policy):
    def __init__(self):
        self.create = True
        self.current_patterns = []  # List of patterns
        self.demands = None  # Current demands (list of products)
        self.sorted_stocks = None  # Sorted stocks by size
        self.sorted_prods = None  # Sorted products by size
        self.num_products = 0
        self.master_solution = None  # Solution from master problem
        self.used_stocks = []
        self.stock_area = 0
        self.prod_area = 0

    def _solve_master_problem(self, demands_vector):
        """
        Solve the master problem using scipy.optimize.linprog.
        """
        num_patterns = len(self.current_patterns)
        cvec = np.ones(num_patterns)

        # Build A_eq matrix
        A_eq = np.zeros((self.num_products, num_patterns))
        for j, pattern in enumerate(self.current_patterns):
            A_eq[:, j] = pattern

        b_eq = demands_vector

        # Solve LP
        result = linprog(cvec, A_eq=A_eq, b_eq=b_eq, method="highs", bounds=(0, None))
        if result.success:
            self.master_solution = result["x"]
        else:
            raise ValueError("Cannot solve master problem.")

    def get_action(self, observation, info):
        # Construct the policy here
        if self.create:
            self.sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=True)
            self.sorted_prods = sorted(enumerate(observation["products"]), key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
            self.num_products = len(self.sorted_prods)   
            self.prod_area = sum([prod[1]["size"][0] * prod[1]["size"][1] * prod[1]["quantity"] for prod in self.sorted_prods])
            for i in range(self.num_products):
                counts = np.zeros(self.num_products, dtype=int)
                counts[i] = 1
                pattern = counts  # Empty placements for now
                self.current_patterns.append(pattern)
            self.create = False
        
        demands_vector = np.array([prod[1]["quantity"] for prod in self.sorted_prods])
        self._solve_master_problem(demands_vector)
        x = self.master_solution
        pattern_idx = np.argmax(x)
        selected_pattern = self.current_patterns[pattern_idx]
        for stock_idx, stock in self.sorted_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)

            for i in range(self.num_products):
                count = selected_pattern[i]
                if count == 0:
                    continue
                product = self.sorted_prods[i][1]
                prod_w, prod_h = product["size"]
                for x_pos in range(stock_w - prod_w + 1):
                    for y_pos in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                            is_used = False
                            for use in self.used_stocks:
                                if use == stock_idx:
                                    is_used = True
                                    break
                            if not is_used:
                                self.stock_area += stock_w * stock_h
                                self.used_stocks.append(stock_idx)
                            return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (x_pos, y_pos)}
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

class mygreed(Policy):
    def __init__(self):
        self.create = True
        self.stock_idx = -1
        self.used_stocks = []
        self.stock_area = 0
        self.prod_area = 0
        self.sorted_prods = None
        self.sorted_stocks = None

    def get_action(self, observation, info):
        if self.create:
            self.sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=True)
            self.sorted_prods = sorted(enumerate(observation["products"]), key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
            self.prod_area = sum([prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in observation["products"]])
            self.create = False
        # Pick a product that has quality > 0
        for _,prod in self.sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                    
                for stock_idx, stock in self.sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    self.stock_idx = stock_idx
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                x_pos, y_pos = x, y
                                is_used = False
                                for use in self.used_stocks:
                                    if use == stock_idx:
                                        is_used = True
                                        break
                                if not is_used:
                                    self.stock_area += stock_w * stock_h
                                    self.used_stocks.append(stock_idx)
                                return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (x_pos, y_pos)}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}   
        
class oneLP(Policy):
    def __init__(self):
        self.create = True
        self.current_patterns = []  # List of patterns
        self.demands = None  # Current demands (list of products)
        self.sorted_stocks = None  # Sorted stocks by size
        self.num_products = 0
        self.master_solution = None  # Solution from master problem
        self.used_stocks = []
        self.stock_area = 0
        self.prod_area = 0

    def _solve_master_problem(self, demands_vector):
        """
        Solve the master problem using scipy.optimize.linprog.
        """
        num_patterns = len(self.current_patterns)
        cvec = np.ones(num_patterns)

        # Build A_eq matrix
        A_eq = np.zeros((self.num_products, num_patterns))
        for j, pattern in enumerate(self.current_patterns):
            A_eq[:, j] = pattern["counts"]

        b_eq = demands_vector

        # Solve LP
        result = linprog(cvec, A_eq=A_eq, b_eq=b_eq, method="highs", bounds=(0, None),integrality=True)
        if result.success:
            self.master_solution = result["x"]
        else:
            raise ValueError("Cannot solve master problem.")

    def get_action(self, observation, info):
        self.demands = observation["products"]
        # Construct the policy here
        if self.create:
            self.prod_area = sum([prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in self.demands])
            self.num_products = len(self.demands)
            self.sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=True)   
            for i in range(self.num_products):
                counts = np.zeros(self.num_products, dtype=int)
                counts[i] = 1
                pattern = {"counts": counts,"sol": 0}  # Empty placements for now
                self.current_patterns.append(pattern)
            demands_vector = np.array([prod["quantity"] for prod in self.demands])
            self._solve_master_problem(demands_vector)
            x = self.master_solution
            for i in range(self.num_products):
                self.current_patterns[i]["sol"] = x[i]
            self.current_patterns = sorted(self.current_patterns, key=lambda x: x["sol"], reverse=True)
            self.create = False
            
        self.current_patterns = [pattern for pattern in self.current_patterns if pattern["sol"] > 0]
        selected_pattern = self.current_patterns[0]["counts"]
        self.current_patterns[0]["sol"] -= 1
        for stock_idx, stock in self.sorted_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            is_used = False
            for use in self.used_stocks:
                if use == stock_idx:
                    is_used = True
                    break
            if not is_used:
                self.stock_area += stock_w * stock_h
                self.used_stocks.append(stock_idx)
            for i in range(self.num_products):
                count = selected_pattern[i]
                if count == 0:
                    continue
                product = self.demands[i]
                prod_w, prod_h = product["size"]
                for x_pos in range(stock_w - prod_w + 1):
                    for y_pos in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                            return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (x_pos, y_pos)}
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}     
  
class myColumn(Policy):
    def __init__(self):
        self.create = True
        self.current_patterns = []  # List of patterns
        self.selected_pattern = None
        self.demands = None  # Current demands (list of products)
        self.sorted_stocks = None  # Sorted stocks by size
        self.num_products = 0
        self.dual_values = None  # Dual values from master problem
        self.master_solution = None  # Solution from master problem
        self.stock_idx = 0
        self.current_stock_size = None
        self.idx = -1
        self.used_stocks = []
        self.stock_area = 0
        self.prod_area = 0

    def _solve_master_problem(self, demands_vector):
        """
        Solve the master problem using scipy.optimize.linprog.
        """
        num_patterns = len(self.current_patterns)
        cvec = np.ones(num_patterns)

        # Build A_eq matrix
        A_eq = np.zeros((self.num_products, num_patterns))
        for j, pattern in enumerate(self.current_patterns):
            A_eq[:, j] = pattern

        b_eq = demands_vector
        # Solve LP
        result = linprog(cvec, A_eq=A_eq, b_eq=b_eq, method="highs", bounds=(0, None))
        if result.success:
            self.master_solution = result["x"]
            self.dual_values = result["eqlin"]["marginals"]
        else:
            raise ValueError("Cannot solve master problem.")

    def create_intial_pattern(self):
        for i in range(self.num_products):
            if self.demands[i]["quantity"] == 0:
                continue
            counts = np.zeros(self.num_products, dtype=int)
            counts[i] = 1
            pattern = counts  # Empty placements for now
            self.current_patterns.append(pattern)

    def create_pattern(self, dual_values = None):
        cvec = dual_values
        if dual_values is None:
            cvec = np.ones(self.num_products)
        cvec *= -1
        A_ub = np.zeros((1, self.num_products))
        b_ub = [self.current_stock_size[0]*self.current_stock_size[1]]
        bounds = [(0, prod["quantity"]) for prod in self.demands]
        A_ub[0, :] = [prod["size"][0] * prod["size"][1]for prod in self.demands]
        result = linprog(cvec, A_ub=A_ub, b_ub=b_ub, method="highs", bounds=bounds,integrality=True)
        self.current_patterns.append(np.int64(result["x"]))

    def return_action(self):
        while True:
            stock = self.sorted_stocks[self.stock_idx][1]
            stock_idx = self.sorted_stocks[self.stock_idx][0]
            stock_w, stock_h = self._get_stock_size_(stock)
            if(self.idx != -1 and self.selected_pattern[self.idx] != 0):
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = self.demands[self.idx]["size"]
                for x_pos in range(stock_w - prod_w + 1):
                    for y_pos in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                            self.selected_pattern[self.idx] -= 1
                            is_used = False
                            for use in self.used_stocks:
                                if use == stock_idx:
                                    is_used = True
                                    break
                            if not is_used:
                                self.stock_area += stock_w * stock_h
                                self.used_stocks.append(stock_idx)
                            return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (x_pos, y_pos)}
            for i in range(self.num_products):
                count = self.selected_pattern[i]
                if count == 0 or self.demands[i]["quantity"] == 0:
                    continue
                self.idx = i
                product = self.demands[i]
                prod_w, prod_h = product["size"]
                for x_pos in range(stock_w - prod_w + 1):
                    for y_pos in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                            self.selected_pattern[i] -= 1
                            is_used = False
                            for use in self.used_stocks:
                                if use == stock_idx:
                                    is_used = True
                                    break
                            if not is_used:
                                self.stock_area += stock_w * stock_h
                                self.used_stocks.append(stock_idx)
                            return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (x_pos, y_pos)}
            self.stock_idx += 1
        
    def get_action(self, observation, info):
        if(self.selected_pattern is not None and np.sum(self.selected_pattern) == 0):
            self.selected_pattern = None
            self.current_patterns = []
        if(self.selected_pattern is not None):
            return self.return_action()
        self.demands = observation["products"]
        # Construct the policy here
        if self.create:
            self.prod_area = sum([prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in self.demands])
            self.num_products = len(self.demands)
            self.sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=True)
            self.create = False
        
        if self.selected_pattern == None:
            self.stock_idx = 0
            self.current_stock_size = self._get_stock_size_(self.sorted_stocks[99][1])
            self.create_intial_pattern()
            demands_vector = np.array([prod["quantity"] for prod in self.demands])
            for i in range(100):
                self._solve_master_problem(demands_vector)
                self.create_pattern(self.dual_values)
            
            x = self.master_solution
            pattern_idx = np.argmax(x)
            self.selected_pattern = self.current_patterns[pattern_idx]
            print(self.selected_pattern) 
            return self.return_action()
        
    
  

class ColumnGenerationPolicy(Policy):
    def __init__(self):
        self.current_patterns = []  # List of patterns
        self.dual_values = None  # Dual values from master problem
        self.demands = None  # Current demands (list of products)
        self.stock_size = None  # Stock size (width, height)
        self.num_products = 0
        self.epsilon = 1e-6  # Tolerance for reduced cost
        self.action_queue = []  # Queue of actions to perform
        self.master_solution = None  # Solution from master problem

    def _solve_master_problem(self, demands_vector):
        """
        Solve the master problem using scipy.optimize.linprog.
        """
        num_patterns = len(self.current_patterns)
        cvec = np.ones(num_patterns)

        # Build A_eq matrix
        A_eq = np.zeros((self.num_products, num_patterns))
        for j, pattern in enumerate(self.current_patterns):
            A_eq[:, j] = pattern['counts']

        b_eq = demands_vector

        # Solve LP
        result = linprog(cvec, A_eq=A_eq, b_eq=b_eq, method="highs", bounds=(0, None))
        if result.success:
            # Store dual values and solution
            self.dual_values = result["eqlin"]["marginals"]
            self.master_solution = result["x"]
        else:
            raise ValueError("Cannot solve master problem.")

    def _solve_subproblem(self):
        """
        Solve the subproblem to generate a new pattern.
        """
        num_products = self.num_products
        stock_w, stock_h = self.stock_size

        # Dual values from master problem
        dual_values = self.dual_values

        # Calculate unit values (e.g., dual value per unit area)
        unit_values = []
        for i in range(num_products):
            product = self.demands[i]
            u_i = dual_values[i]
            w_i, h_i = product["size"]
            area_i = w_i * h_i
            unit_value = u_i / area_i if area_i > 0 else 0
            unit_values.append((unit_value, i))

        # Sort products by unit value
        unit_values.sort(reverse=True)

        # Initialize empty stock
        stock = np.full((stock_w, stock_h), fill_value=-1, dtype=int)

        counts = np.zeros(num_products, dtype=int)
        placements = []

        # Try to place products
        # for unit_value, i in unit_values:
        #     product = self.demands[i]
        #     w_i, h_i = product["size"]
        #     quantity_i = product["quantity"]

        #     # For each available quantity
        #     for _ in range(quantity_i - counts[i]):
        #         placed = False
        #         for x in range(stock_w - w_i + 1):
        #             for y in range(stock_h - h_i + 1):
        #                 if np.all(stock[x:x + w_i, y:y + h_i] == -1):
        #                     # Place the product
        #                     stock[x:x + w_i, y:y + h_i] = i
        #                     counts[i] += 1
        #                     placements.append((i, x, y, w_i, h_i))
        #                     placed = True
        #                     break
        #             if placed:
        #                 break
        #         if not placed:
        #             # Cannot place more of this product
        #             break

        # Calculate reduced cost
        reduced_cost = 1 - np.dot(self.dual_values, counts)

        if reduced_cost < -self.epsilon:
            # Return new pattern
            pattern = {
                'counts': counts,
                'placements': placements
            }
            return pattern
        else:
            return None

    def get_action(self, observation, info):
        # If we have actions in queue, return the next one
        if self.action_queue:
            return self.action_queue.pop(0)

        # Update demands from observation
        self.demands = observation["products"]
        self.num_products = len(self.demands)

        # Update stock size
        self.stock_size = self._get_stock_size_(observation["stocks"][0])  # Assume stocks are same size
        # print("Stock size:", self.stock_size)
        # Prepare demands vector
        demands_vector = np.array([prod["quantity"] for prod in self.demands])

        # Initialize current_patterns if empty
        if not self.current_patterns:
            # Initialize with unit patterns (one product per pattern)
            for i in range(self.num_products):
                counts = np.zeros(self.num_products, dtype=int)
                counts[i] = 1
                pattern = {'counts': counts, 'placements': []}  # Empty placements for now
                self.current_patterns.append(pattern)
        
        # Column generation loop
        while True:
            # Solve master problem
            self._solve_master_problem(demands_vector)

            # Solve subproblem
            new_pattern = self._solve_subproblem()

            if new_pattern is None:
                # No more patterns to add
                break

            # Add new pattern
            self.current_patterns.append(new_pattern)

        # From master problem solution, get x_j
        x = self.master_solution
        # Select the pattern with highest x_j
        pattern_idx = np.argmax(x)
        selected_pattern = self.current_patterns[pattern_idx]
        # print("Selected pattern:", selected_pattern)
        # Now, try to apply the selected pattern to a stock
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Initialize a stock grid
            stock_grid = stock.copy()

            # Try to place the products as per the pattern placements
            placements = []
            success = True

            for i in range(self.num_products):
                count = selected_pattern['counts'][i]
                if count == 0:
                    continue
                product = self.demands[i]
                prod_w, prod_h = product["size"]
                quantity = min(count, product["quantity"])

                for _ in range(quantity):
                    placed = False
                    for x_pos in range(stock_w - prod_w + 1):
                        for y_pos in range(stock_h - prod_h + 1):
                            if self._can_place_(stock_grid, (x_pos, y_pos), (prod_w, prod_h)):
                                # Place the product
                                stock_grid[x_pos:x_pos + prod_w, y_pos:y_pos + prod_h] = i
                                placements.append((i, x_pos, y_pos, prod_w, prod_h))
                                placed = True
                                break
                        if placed:
                            break
                    if not placed:
                        success = False
                        break  # Cannot place this product
                if not success:
                    break  # Cannot place products as per pattern
            # print("Placements:", placements)
            if success:
                # We have found a stock where we can apply the pattern
                # Create actions for the placements
                self.action_queue = []
                for placement in placements:
                    i, x_pos, y_pos, prod_w, prod_h = placement
                    action = {
                        'stock_idx': stock_idx,
                        'size': [prod_w, prod_h],
                        'position': (x_pos, y_pos)
                    }
                    self.action_queue.append(action)
                # Return the first action
                return self.action_queue.pop(0)
        # If we cannot apply the pattern, return default action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}