from policy import Policy

##My policy
class Policy2210xxx(Policy):
    def __init__(self):
        # Student code here
        #pass
        self.columns = [] #Sử dụng columns generation

    def generate_columns(self, stock, prod_size):
        # Student code here
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        #Check if the stock still can store the product
        if stock_w < prod_w or stock_h < prod_h:
            return None
        
        #Create a suitable cutting pattern
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    #If find a suitable cutting pattern, add to the columns
                    column = {"position": (x, y), "size": prod_size}
                    self.columns.append(column)
                    return column
        return None


    def get_action(self, observation, info):
        # Student code here
        #pass
        list_prods = observation["products"]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        selected_column = None

        #Traverse through all products have quantity > 0 to find cutting patterns
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                #Create new cutting pattern from remaining stock
                for i, stock in enumerate(observation["stocks"]):
                    column = self.generate_columns(stock, prod_size)
                    if column is not None:
                        stock_idx = i
                        pos_x, pos_y = column["position"]
                        selected_column = column
                        break
                if selected_column is not None:
                    break

        #If not find any suitable cutting pattern, return default action
        if selected_column is None:
            return {"stock_idx": stock_idx, "size": [0, 0], "position": (0, 0)}
        
        #Return the selected cutting pattern
        return {"stock_idx": stock_idx, "size": selected_column["size"], "position": (pos_x, pos_y)}


class ISHP_Policy(Policy):
    def __init__(self, max_iterations=100, alpha=0.7, beta=1.02):
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.best_cutting_plan = None
        self.best_plate_count = float('inf')

    def generate_2SGP(self, stock, prod_size):
        # Generate two-staged general patterns (2SGP) based on prod_size and stock constraints
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return None

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return {"position": (x, y), "size": prod_size}
        return None

    def generate_3SHP(self, stock, prod_size):
        # Generate three-staged homogeneous patterns (3SHP) based on prod_size and stock constraints
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return None

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return {"position": (x, y), "size": prod_size}
        return None

    def select_best_pattern(self, stock, prod_size):
        pattern_2SGP = self.generate_2SGP(stock, prod_size)
        pattern_3SHP = self.generate_3SHP(stock, prod_size)

        if pattern_2SGP and pattern_3SHP:
            # Calculate the area of each pattern's size
            area_2SGP = pattern_2SGP["size"][0] * pattern_2SGP["size"][1]
            area_3SHP = pattern_3SHP["size"][0] * pattern_3SHP["size"][1]
            return pattern_2SGP if area_2SGP > area_3SHP else pattern_3SHP
        return pattern_2SGP if pattern_2SGP else pattern_3SHP


    def get_action(self, observation, info):
        list_prods = observation["products"]
        for iteration in range(self.max_iterations):
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]

                    for i, stock in enumerate(observation["stocks"]):
                        candidate_pattern = self.select_best_pattern(stock, prod_size)
                        if candidate_pattern:
                            prod["quantity"] -= 1
                            stock_idx = i
                            pos_x, pos_y = candidate_pattern["position"]

                            return {
                                "stock_idx": stock_idx,
                                "size": candidate_pattern["size"],
                                "position": (pos_x, pos_y)
                            }
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    

class ApproximationPolicy(Policy):
    def __init__(self, max_iterations=100):
        self.max_iterations = max_iterations

    def generate_initial_patterns(self, stock, prod_size):
        """Generate an initial set of cutting patterns for the knapsack problem."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Ensure product can fit within stock dimensions
        if stock_w < prod_w or stock_h < prod_h:
            return None

        patterns = []
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    patterns.append({"position": (x, y), "size": prod_size})
        return patterns

    def solve_knapsack_problem(self, patterns, stock):
        """Use dynamic programming to solve the knapsack problem for cutting patterns."""
        best_value = 0
        best_pattern = None

        for pattern in patterns:
            pos_x, pos_y = pattern["position"]
            size_w, size_h = pattern["size"]

            # Calculate value based on area used
            value = size_w * size_h  # Alternatively, customize this value metric
            if value > best_value:
                best_value = value
                best_pattern = pattern

        return best_pattern

    def get_action(self, observation, info):
        list_prods = observation["products"]

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    # Generate initial cutting patterns
                    patterns = self.generate_initial_patterns(stock, prod_size)
                    if not patterns:
                        continue

                    # Solve the knapsack problem for the generated patterns
                    best_pattern = self.solve_knapsack_problem(patterns, stock)
                    if best_pattern:
                        # Update the quantity and return the action
                        prod["quantity"] -= 1
                        return {
                            "stock_idx": i,
                            "size": best_pattern["size"],
                            "position": best_pattern["position"]
                        }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    



class KenyonRemilaPolicy(Policy):
    def __init__(self, threshold_ratio=0.2, group_count=5):
        """
        Initialize the Kenyon-Remila Policy.
        :param threshold_ratio: Ratio to determine narrow vs wide rectangles.
        :param group_count: Number of groups for wide rectangles.
        """
        self.threshold_ratio = threshold_ratio
        self.group_count = group_count

    def _partition_rectangles(self, products):
        """
        Partition rectangles into narrow and wide groups based on threshold ratio.
        :param products: List of product dictionaries.
        :return: narrow_rectangles, wide_rectangles
        """
        narrow = []
        wide = []
        for prod in products:
            if prod["quantity"] > 0:
                width, height = prod["size"]
                if width / height < self.threshold_ratio:
                    narrow.append(prod)
                else:
                    wide.append(prod)
        return narrow, wide

    def _place_rectangle(self, stock, rect_size):
        """
        Place a rectangle in the stock using the Best Fit strategy.
        :param stock: 2D stock array.
        :param rect_size: Size of the rectangle to place.
        :return: Position (x, y) if successful, None otherwise.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        rect_w, rect_h = rect_size

        # Check if the rectangle fits in the stock
        best_position = None
        best_waste = float('inf')

        for x in range(stock_w - rect_w + 1):
            for y in range(stock_h - rect_h + 1):
                if self._can_place_(stock, (x, y), (rect_w, rect_h)):
                    # Calculate wasted space if placed here
                    waste = (stock_w - rect_w) * (stock_h - rect_h)
                    if waste < best_waste:
                        best_position = (x, y)
                        best_waste = waste

        return best_position

    def get_action(self, observation, info):
        """
        Implement the heuristic to get the next action.
        :param observation: Current environment observation.
        :param info: Additional information from the environment.
        :return: Action dictionary.
        """
        products = observation["products"]
        stocks = observation["stocks"]

        # Step 1: Partition rectangles into narrow and wide groups
        narrow, wide = self._partition_rectangles(products)

        # Step 2: Attempt to place wide rectangles first
        for rect in wide:
            for stock_idx, stock in enumerate(stocks):
                position = self._place_rectangle(stock, rect["size"])
                if position:
                    rect["quantity"] -= 1
                    return {
                        "stock_idx": stock_idx,
                        "size": rect["size"],
                        "position": position,
                    }

        # Step 3: Attempt to place narrow rectangles using NFDH heuristic
        for rect in narrow:
            for stock_idx, stock in enumerate(stocks):
                position = self._place_rectangle(stock, rect["size"])
                if position:
                    rect["quantity"] -= 1
                    return {
                        "stock_idx": stock_idx,
                        "size": rect["size"],
                        "position": position,
                    }

        # If no valid placement is found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


class BendersDecompositionPolicy(Policy):
    def __init__(self, max_iterations=100):
        """
        Initialize the policy with the maximum iterations and other relevant parameters.
        """
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_filled_ratio = 0

    def generate_initial_patterns(self, stock, prod_size):
        """
        Generate initial patterns for potential placement of the product.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return None

        patterns = []
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    patterns.append({"position": (x, y), "size": prod_size})
        return patterns

    def solve_master_problem(self, patterns):
        """
        Solve the master problem by choosing the best pattern.
        """
        best_pattern = None
        best_value = 0

        for pattern in patterns:
            area = pattern["size"][0] * pattern["size"][1]
            if area > best_value:
                best_value = area
                best_pattern = pattern

        return best_pattern

    def get_action(self, observation, info):
        """
        Use Benders' decomposition to find an optimal cutting pattern.
        """
        list_prods = observation["products"]

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    # Generate potential patterns
                    patterns = self.generate_initial_patterns(stock, prod_size)
                    if not patterns:
                        continue

                    # Solve the master problem
                    best_pattern = self.solve_master_problem(patterns)
                    if best_pattern:
                        # Update quantities and return action
                        prod["quantity"] -= 1
                        return {
                            "stock_idx": i,
                            "size": best_pattern["size"],
                            "position": best_pattern["position"]
                        }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

