from PIL import Image
import cv2
import torch
from typing import Optional
from transformers import CLIPProcessor, CLIPModel

import sys

sys.path.append("../")
from utils import read_stub, save_stub


class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    The class uses a pre-trained vision model to classify players into teams based on their
    appearance. It maintains a consistent team assignment for each player across frames.

    Attributes:
        team_colors (dict): Dictionary storing team color information.
        player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
    """

    def __init__(
        self,
        team_1_class_name: str = "white shirt",
        team_2_class_name: str = "dark blue shirt",
        device: Optional[str] = None,
        use_half: bool = False,
    ):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args:
            team_1_class_name (str): Description of Team 1's jersey appearance.
            team_2_class_name (str): Description of Team 2's jersey appearance.
        """
        self.team_colors = {}
        self.player_team_dict = {}

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

        # Device handling (CPU/GPU)
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_half = bool(use_half and self.device.type == "cuda")

    def load_model(self):
        """
        Loads the pre-trained vision model for jersey color classification (GPU-aware).
        """
        dtype = torch.float16 if self.use_half else torch.float32
        self.model = CLIPModel.from_pretrained(
            "patrickjohncyh/fashion-clip", torch_dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        """
        Analyzes the jersey color of a player within the given bounding box.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            bbox (tuple): Bounding box coordinates of the player.

        Returns:
            str: The classified jersey color/description.
        """
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Convert to PIL Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = pil_image

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(
            text=classes, images=image, return_tensors="pt", padding=True
        )
        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape: [1, num_texts]
            probs = logits_per_image.softmax(dim=1)

        pred_idx = int(probs.argmax(dim=1).item())
        class_name = classes[pred_idx]

        return class_name

    def get_players_color_batch(self, frame, items):
        """
        Batch inference for a list of (player_id, bbox) items on a single frame.
        Returns: dict player_id -> class_name
        """
        if not items:
            return {}
        crops = []
        ids = []
        for pid, bbox in items:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            img = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb))
            ids.append(pid)

        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(text=classes, images=crops, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # [N, 2]
            probs = logits.softmax(dim=1)
            pred = probs.argmax(dim=1).tolist()

        result = {}
        for pid, idx in zip(ids, pred):
            result[pid] = classes[int(idx)]
        return result

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Gets the team assignment for a player, using cached results if available.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            player_bbox (tuple): Bounding box coordinates of the player.
            player_id (int): Unique identifier for the player.

        Returns:
            int: Team ID (1 or 2) assigned to the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(
        self,
        video_frames,
        player_tracks,
        read_from_stub: bool = False,
        stub_path: str | None = None,
        stride: int = 1,
        batch: bool = False,
    ):
        """
        Processes all video frames to assign teams to players, with optional caching.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of player tracking information for each frame.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries mapping player IDs to team assignments for each frame.
        """

        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        player_assignment = []

        # Process each frame
        for frame_num in range(len(video_frames)):
            # initialize mapping per frame
            if frame_num == 0:
                player_assignment.append({})
            else:
                # by default, carry over previous assignments
                player_assignment.append(dict(player_assignment[frame_num - 1]))

            if frame_num % 50 == 0:
                self.player_team_dict = {}

            # Get player tracks for this frame and ensure it's a dictionary
            if frame_num in player_tracks:
                player_track = player_tracks[frame_num]
                if isinstance(player_track, dict):
                    # Sampling: only recompute on stride frames, otherwise reuse mapping
                    if stride <= 1 or frame_num % max(1, stride) == 0:
                        if batch:
                            items = [
                                (pid, trk["bbox"]) for pid, trk in player_track.items() if trk.get("bbox")
                            ]
                            batch_map = self.get_players_color_batch(video_frames[frame_num], items)
                            for player_id, track in player_track.items():
                                cls_name = batch_map.get(player_id)
                                if cls_name is None:
                                    cls_name = self.get_player_color(
                                        video_frames[frame_num], track["bbox"]
                                    )
                                team_id = 1 if cls_name == self.team_1_class_name else 2
                                self.player_team_dict[player_id] = team_id
                                player_assignment[frame_num][player_id] = team_id
                        else:
                            for player_id, track in player_track.items():
                                team = self.get_player_team(
                                    video_frames[frame_num], track["bbox"], player_id
                                )
                                player_assignment[frame_num][player_id] = team
                else:
                    print(
                        f"Warning: Expected dictionary for player_track at frame {frame_num}, but got {type(player_track).__name__}"
                    )

        save_stub(stub_path, player_assignment)

        return player_assignment
