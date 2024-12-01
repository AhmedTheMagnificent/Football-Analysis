import sys
sys.path.append("../")
from utils import get_center_of_box, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        
    def assign_ball_to_player(self, players, ball_box):
        ball_positions = get_center_of_box(ball_box)
        
        if not ball_positions:
            return -1
        
        
        minimum_distance = float("inf")
        assigned_player = -1  # Default to -1, indicating no player assigned
        
        for player_id, player in players.items():
            player_box = player["box"]
            
            distance_left = measure_distance((player_box[0], player_box[-1]), ball_positions)
            distance_right = measure_distance((player_box[2], player_box[-1]), ball_positions)
            distance = min(distance_left, distance_right)
            
            
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id
        
        
        return assigned_player
