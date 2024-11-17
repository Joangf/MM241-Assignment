from policy import Policy


def Policy2210xxx(Policy):
    def __init__(self):
        # Student code here
        pass

    def get_action(self, observation, info):
        # Student code here
        pass

    # Student code here
    # You can add more functions if needed

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