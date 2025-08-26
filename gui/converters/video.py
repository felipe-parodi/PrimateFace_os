"""Video to COCO format converter.

This module provides functionality to convert videos to COCO annotation
format by extracting frames and processing them through detection and
pose estimation pipelines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
from tqdm import tqdm

from ..utils import extract_frames_from_video, parallel_process_videos
from .base import COCOConverter


class VideoCOCOConverter(COCOConverter):
    """Converter for videos to COCO format.
    
    This class processes video files by extracting frames at specified
    intervals and running detection/pose estimation on them.
    
    Attributes:
        frame_interval: Extract every Nth frame.
        save_frames: Whether to save extracted frames.
        video_extensions: Supported video file extensions.
    """
    
    def __init__(
        self,
        *args,
        frame_interval: int = 30,
        save_frames: bool = True,
        video_extensions: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Initialize VideoCOCOConverter.
        
        Args:
            *args: Arguments for COCOConverter.
            frame_interval: Extract every Nth frame.
            save_frames: Whether to save extracted frames.
            video_extensions: List of video extensions to process.
            **kwargs: Keyword arguments for COCOConverter.
        """
        super().__init__(*args, **kwargs)
        self.frame_interval = frame_interval
        self.save_frames = save_frames
        
        if video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        else:
            self.video_extensions = video_extensions
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_json: Optional[Union[str, Path]] = None,
        frames_output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None
    ) -> str:
        """Process a single video file.
        
        Args:
            video_path: Path to video file.
            output_json: Output path for COCO JSON.
            frames_output_dir: Directory to save extracted frames.
            max_frames: Maximum number of frames to process.
            
        Returns:
            Path to saved COCO JSON file.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file does not exist: {video_path}")
        
        if frames_output_dir is None:
            frames_output_dir = self.output_dir / "frames" / video_path.stem
        else:
            frames_output_dir = Path(frames_output_dir)
        
        frames_output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing video: {video_path.name}")
        print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
        print(f"Processing every {self.frame_interval} frames")
        
        all_images = []
        all_annotations = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_interval == 0:
                frame_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
                
                if self.save_frames:
                    frame_path = frames_output_dir / frame_name
                    cv2.imwrite(str(frame_path), frame)
                
                image_info, annotations, _ = self.process_image(
                    frame,
                    frame_name,
                    return_visualizations=False
                )
                
                all_images.append(image_info)
                all_annotations.extend(annotations)
            
            frame_idx += 1
            pbar.update(1)
            
            if max_frames and frame_idx >= max_frames:
                break
        
        cap.release()
        pbar.close()
        
        coco_data = self.create_coco_dataset(all_images, all_annotations)
        
        if output_json is None:
            output_json = self.output_dir / f"{video_path.stem}_annotations.json"
        
        return self.save_coco_json(coco_data, output_json)
    
    def process_directory(
        self,
        video_dir: Union[str, Path],
        output_json: Optional[Union[str, Path]] = None,
        frames_output_dir: Optional[Union[str, Path]] = None,
        max_videos: Optional[int] = None,
        max_frames_per_video: Optional[int] = None
    ) -> str:
        """Process all videos in a directory.
        
        Args:
            video_dir: Directory containing videos.
            output_json: Output path for COCO JSON.
            frames_output_dir: Directory to save extracted frames.
            max_videos: Maximum number of videos to process.
            max_frames_per_video: Maximum frames per video.
            
        Returns:
            Path to saved COCO JSON file.
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise ValueError(f"Video directory does not exist: {video_dir}")
        
        video_paths = self._collect_video_paths(video_dir)
        
        if not video_paths:
            raise ValueError(f"No videos found in {video_dir}")
        
        if max_videos:
            video_paths = video_paths[:max_videos]
        
        print(f"Processing {len(video_paths)} videos...")
        
        if frames_output_dir is None:
            frames_output_dir = self.output_dir / "frames"
        else:
            frames_output_dir = Path(frames_output_dir)
        
        all_images = []
        all_annotations = []
        
        for video_path in video_paths:
            video_frames_dir = frames_output_dir / video_path.stem
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            
            temp_json = self.output_dir / f"temp_{video_path.stem}.json"
            
            self.reset_counters()
            json_path = self.process_video(
                video_path,
                temp_json,
                video_frames_dir,
                max_frames_per_video
            )
            
            import json
            with open(json_path, 'r') as f:
                video_data = json.load(f)
            
            all_images.extend(video_data["images"])
            all_annotations.extend(video_data["annotations"])
            
            temp_json.unlink()
        
        coco_data = self.create_coco_dataset(all_images, all_annotations)
        
        if output_json is None:
            output_json = self.output_dir / "annotations.json"
        
        return self.save_coco_json(coco_data, output_json)
    
    def _collect_video_paths(
        self,
        directory: Path
    ) -> List[Path]:
        """Collect all video paths from directory.
        
        Args:
            directory: Directory to search.
            
        Returns:
            List of video file paths.
        """
        video_paths = []
        
        for ext in self.video_extensions:
            video_paths.extend(directory.glob(f"*{ext}"))
            video_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(list(set(video_paths)))
    
    def process_parallel(
        self,
        video_dir: Union[str, Path],
        output_json: Optional[Union[str, Path]] = None,
        gpus: Optional[List[int]] = None,
        jobs_per_gpu: int = 1
    ) -> str:
        """Process videos in parallel across multiple GPUs.
        
        Args:
            video_dir: Directory containing videos.
            output_json: Output path for COCO JSON.
            gpus: List of GPU IDs to use.
            jobs_per_gpu: Number of parallel jobs per GPU.
            
        Returns:
            Path to saved COCO JSON file.
        """
        if gpus is None:
            gpus = [0]
        
        def process_func(args, video_files):
            results = {"images": [], "annotations": [], "categories": self.categories}
            
            for video_path in video_files:
                converter = VideoCOCOConverter(
                    self.detector,
                    self.pose_estimator,
                    self.sam_masker,
                    output_dir=args.output_dir,
                    frame_interval=self.frame_interval,
                    save_frames=self.save_frames
                )
                
                converter.device = args.device
                
                temp_json = Path(args.output_dir) / f"temp_{video_path.stem}.json"
                json_path = converter.process_video(video_path, temp_json)
                
                import json
                with open(json_path, 'r') as f:
                    video_data = json.load(f)
                
                results["images"].extend(video_data["images"])
                results["annotations"].extend(video_data["annotations"])
                
                temp_json.unlink()
            
            return results
        
        class Args:
            def __init__(self, output_dir, device):
                self.output_dir = output_dir
                self.device = device
        
        args = Args(str(self.output_dir), "cuda:0")
        
        return parallel_process_videos(
            video_dir,
            self.output_dir,
            process_func,
            args,
            gpus,
            jobs_per_gpu
        )