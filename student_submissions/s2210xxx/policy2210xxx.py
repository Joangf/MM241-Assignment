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

class GraphBasedPolicy(Policy):
    def __init__(self):
        """
        Initialize the Graph-Based policy.
        """
        self.graph = nx.DiGraph()

    def build_graph(self, stocks, products):
        """
        Build a graph representing the feasible placements of products in stocks.
        :param stocks: List of stocks.
        :param products: List of products.
        """
        self.graph.clear()

        # Add nodes for products and stocks
        for stock_idx, stock in enumerate(stocks):
            self.graph.add_node(f"stock_{stock_idx}", type="stock")

        for product_idx, product in enumerate(products):
            if product["quantity"] > 0:
                self.graph.add_node(f"product_{product_idx}", type="product", size=product["size"])

        # Add edges representing feasible placements
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            for product_idx, product in enumerate(products):
                if product["quantity"] > 0:
                    prod_w, prod_h = product["size"]

                    # Check all possible placements in the stock
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                self.graph.add_edge(
                                    f"product_{product_idx}",
                                    f"stock_{stock_idx}",
                                    position=(x, y),
                                    size=(prod_w, prod_h)
                                )

    def find_best_placement(self):
        """
        Find the best placement by traversing the graph.
        :return: Best placement as a tuple (product_node, stock_node, position, size).
        """
        # Prioritize placing larger products first
        product_nodes = [n for n, attr in self.graph.nodes(data=True) if attr["type"] == "product"]
        product_nodes.sort(key=lambda n: np.prod(self.graph.nodes[n]["size"]), reverse=True)

        for product_node in product_nodes:
            # Get all possible placements for the product
            edges = self.graph.edges(product_node, data=True)
            for _, stock_node, edge_data in edges:  # Unpack correctly
                # Return the first feasible placement
                return product_node, stock_node, edge_data["position"], edge_data["size"]

        return None


    def get_action(self, observation, info):
        """
        Use the graph-based approach to determine the next action.
        :param observation: Current environment observation.
        :param info: Additional information from the environment.
        :return: Action dictionary for cutting stock.
        """
        stocks = observation["stocks"]
        products = observation["products"]

        # Build the graph
        self.build_graph(stocks, products)

        # Find the best placement
        best_placement = self.find_best_placement()

        if best_placement:
            product_node, stock_node, position, size = best_placement

            # Update the product quantity
            product_idx = int(product_node.split("_")[1])
            stock_idx = int(stock_node.split("_")[1])
            products[product_idx]["quantity"] -= 1

            return {
                "stock_idx": stock_idx,
                "size": size,
                "position": position
            }

        # No valid placement found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


class ConstraintProgrammingPolicy(Policy):
    def __init__(self, max_iterations=100):
        """
        Initialize the Constraint Programming policy.
        :param max_iterations: Maximum iterations to explore feasible solutions.
        """
        self.max_iterations = max_iterations

    def is_feasible(self, stock, product, position):
        """
        Check if a product can be placed in the given stock at the specified position.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product

        x, y = position
        if x + prod_w > stock_w or y + prod_h > stock_h:
            return False

        return self._can_place_(stock, position, product)

    def create_constraints(self, stocks, products):
        """
        Create constraints for the placement of products in stocks.
        :param stocks: List of stocks.
        :param products: List of products.
        :return: List of feasible placements satisfying the constraints.
        """
        constraints = []

        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            for product_idx, product in enumerate(products):
                if product["quantity"] > 0:
                    prod_w, prod_h = product["size"]

                    # Generate all possible positions
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self.is_feasible(stock, (prod_w, prod_h), (x, y)):
                                constraints.append({
                                    "stock_idx": stock_idx,
                                    "product_idx": product_idx,
                                    "position": (x, y),
                                    "size": (prod_w, prod_h)
                                })

        return constraints

    def solve_constraints(self, constraints, stocks, products):
        """
        Solve the constraints using a greedy approach to find the optimal placement.
        :param constraints: List of constraints.
        :param stocks: List of stocks.
        :param products: List of products.
        :return: Optimal placement or None if no solution exists.
        """
        # Sort constraints by product size (larger first) to maximize stock utilization
        constraints.sort(key=lambda c: c["size"][0] * c["size"][1], reverse=True)

        for constraint in constraints:
            stock_idx = constraint["stock_idx"]
            product_idx = constraint["product_idx"]
            position = constraint["position"]
            size = constraint["size"]

            # Check if the placement is still valid
            if products[product_idx]["quantity"] > 0 and self.is_feasible(stocks[stock_idx], size, position):
                # Place the product
                return constraint

        return None

    def get_action(self, observation, info):
        """
        Use Constraint Programming to determine the next action.
        :param observation: Current environment observation.
        :param info: Additional information from the environment.
        :return: Action dictionary for cutting stock.
        """
        stocks = observation["stocks"]
        products = observation["products"]

        # Step 1: Create constraints
        constraints = self.create_constraints(stocks, products)

        # Step 2: Solve constraints
        best_constraint = self.solve_constraints(constraints, stocks, products)

        if best_constraint:
            # Update product quantity and return action
            product_idx = best_constraint["product_idx"]
            products[product_idx]["quantity"] -= 1

            return {
                "stock_idx": best_constraint["stock_idx"],
                "size": best_constraint["size"],
                "position": best_constraint["position"]
            }

        # No valid placement found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}