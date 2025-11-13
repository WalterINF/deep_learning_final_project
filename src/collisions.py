import math

class Raycast:
    origin_x: float
    origin_y: float
    theta: float
    length: float

    def __init__(self, origin_x: float, origin_y: float, theta: float, length: float):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.theta = theta
        self.length = length

    def check_collision(self, other: 'BoundingBox') -> float | None:
        """ Verifica se há colisão entre esta raycast e uma bounding box
        Args:
            other: BoundingBox a ser comparada
        Returns:
            float | None: Distância da origem até o ponto de colisão, ou None se não houver colisão
        """
        # Transform ray origin to box's local coordinate system
        # First translate to box center, then rotate by -theta
        dx = self.origin_x - other.position_x
        dy = self.origin_y - other.position_y
        
        cos_theta = math.cos(-other.theta)
        sin_theta = math.sin(-other.theta)
        
        # Rotate ray origin to box's local space
        local_origin_x = dx * cos_theta - dy * sin_theta
        local_origin_y = dx * sin_theta + dy * cos_theta
        
        # Transform ray direction to box's local space
        ray_dir_x = math.cos(self.theta)
        ray_dir_y = math.sin(self.theta)
        
        local_dir_x = ray_dir_x * cos_theta - ray_dir_y * sin_theta
        local_dir_y = ray_dir_x * sin_theta + ray_dir_y * cos_theta
        
        # Box bounds in local space (axis-aligned)
        half_w = other.width / 2.0
        half_h = other.height / 2.0
        
        box_min_x = -half_w
        box_max_x = half_w
        box_min_y = -half_h
        box_max_y = half_h
        
        # Ray-AABB intersection using slab method
        # Calculate intersection distances for each axis
        if abs(local_dir_x) < 1e-9:  # Ray parallel to y-axis
            if local_origin_x < box_min_x or local_origin_x > box_max_x:
                return None
            # Ray is inside box bounds on x-axis, check y-axis
            if abs(local_dir_y) < 1e-9:
                # Ray is a point, treat as no valid direction
                return None
            
            t1 = (box_min_y - local_origin_y) / local_dir_y if local_dir_y != 0 else float('inf')
            t2 = (box_max_y - local_origin_y) / local_dir_y if local_dir_y != 0 else float('inf')
            t_min = min(t1, t2)
            t_max = max(t1, t2)
        elif abs(local_dir_y) < 1e-9:  # Ray parallel to x-axis
            if local_origin_y < box_min_y or local_origin_y > box_max_y:
                return None
            # Ray is inside box bounds on y-axis, check x-axis
            t1 = (box_min_x - local_origin_x) / local_dir_x if local_dir_x != 0 else float('inf')
            t2 = (box_max_x - local_origin_x) / local_dir_x if local_dir_x != 0 else float('inf')
            t_min = min(t1, t2)
            t_max = max(t1, t2)
        else:
            # Calculate intersection distances for x-axis
            t_x1 = (box_min_x - local_origin_x) / local_dir_x
            t_x2 = (box_max_x - local_origin_x) / local_dir_x
            t_x_min = min(t_x1, t_x2)
            t_x_max = max(t_x1, t_x2)
            
            # Calculate intersection distances for y-axis
            t_y1 = (box_min_y - local_origin_y) / local_dir_y
            t_y2 = (box_max_y - local_origin_y) / local_dir_y
            t_y_min = min(t_y1, t_y2)
            t_y_max = max(t_y1, t_y2)
            
            # Find the intersection range
            t_min = max(t_x_min, t_y_min)
            t_max = min(t_x_max, t_y_max)
        
        # Check if there's no intersection
        if t_min > t_max or t_max < 0:
            return None

        # If outside, first hit forward is t_min (>= 0).
        # If inside (t_min < 0 <= t_max), first forward hit is the exit t_max.
        t_intersection = t_min if t_min >= 0 else t_max

        # Check if intersection is within ray length and forward
        if t_intersection < 0 or t_intersection > self.length:
            return None

        # Return the distance from origin to intersection point
        return t_intersection

class BoundingBox:

    position_x: float
    position_y: float
    width: float
    height: float
    theta: float # ângulo em relação ao eixo x do mapa

    def __init__(self, position_x: float, position_y: float, width: float, height: float, theta: float):
        self.position_x = position_x
        self.position_y = position_y
        self.width = width
        self.height = height
        self.theta = theta

    def check_collision(self, other: 'BoundingBox') -> bool:
        """ Verifica se há colisão entre duas bounding boxes 
        Args:
            other: BoundingBox a ser comparada
        Returns:
            bool: True se houver colisão, False caso contrário
        """
        # Oriented Bounding Box (OBB) intersection via Separating Axis Theorem (SAT)
        # Local axes for each box (unit vectors)
        cos1 = math.cos(self.theta)
        sin1 = math.sin(self.theta)
        cos2 = math.cos(other.theta)
        sin2 = math.sin(other.theta)

        # Axes for self
        u1 = (cos1, sin1)          # axis along width
        v1 = (-sin1, cos1)         # axis along height
        # Axes for other
        u2 = (cos2, sin2)
        v2 = (-sin2, cos2)

        half_w1 = self.width / 2.0
        half_h1 = self.height / 2.0
        half_w2 = other.width / 2.0
        half_h2 = other.height / 2.0

        # Vector between centers
        center_dx = other.position_x - self.position_x
        center_dy = other.position_y - self.position_y

        def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
            return a[0] * b[0] + a[1] * b[1]

        axes = [u1, v1, u2, v2]
        eps = 1e-9

        for axis in axes:
            # Projection radii of each box onto current axis
            r1 = half_w1 * abs(dot(u1, axis)) + half_h1 * abs(dot(v1, axis))
            r2 = half_w2 * abs(dot(u2, axis)) + half_h2 * abs(dot(v2, axis))

            # Distance between centers projected onto axis
            dist = abs(center_dx * axis[0] + center_dy * axis[1])

            # If there is a separating axis, boxes do not intersect
            if dist > (r1 + r2) + eps:
                return False

        # No separating axis found -> boxes intersect
        return True

    def contains_point(self, point: tuple[float, float]) -> bool:
        """verifica se um ponto está dentro da bounding box"""
        # Transform point to this box's local coordinates (translate, then rotate by -theta)
        dx = point[0] - self.position_x
        dy = point[1] - self.position_y

        cos_theta = math.cos(-self.theta)
        sin_theta = math.sin(-self.theta)

        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta

        half_w = self.width / 2.0
        half_h = self.height / 2.0
        eps = 1e-9

        return (-half_w - eps) <= local_x <= (half_w + eps) and (-half_h - eps) <= local_y <= (half_h + eps)

    def contains_bounding_box(self, other: 'BoundingBox') -> bool:
        """verifica se uma bounding box está dentro da bounding box"""
        # All 4 corners of the other OBB must be inside this OBB
        for corner in other.get_corners():
            if not self.contains_point(corner):
                return False
        return True
    
    def get_corners(self) -> list[tuple[float, float]]:
        """Calcula os quatro cantos da bounding box em coordenadas globais"""
        # Half dimensions
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        
        # Local corners (relative to center, before rotation)
        local_corners = [
            (-half_w, -half_h),  # bottom-left
            (half_w, -half_h),   # bottom-right
            (half_w, half_h),    # top-right
            (-half_w, half_h)    # top-left
        ]
        
        # Rotate and translate to global coordinates
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        corners = []
        for local_x, local_y in local_corners:
            # Rotate
            rotated_x = local_x * cos_theta - local_y * sin_theta
            rotated_y = local_x * sin_theta + local_y * cos_theta
            # Translate
            global_x = rotated_x + self.position_x
            global_y = rotated_y + self.position_y
            corners.append((global_x, global_y))
        
        return corners
