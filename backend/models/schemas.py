from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    video_id: str = Field(..., description="Unique YouTube Video ID")
    title: str = Field(..., description="Full lecture title")
    url: str = Field(..., description="Direct YouTube video URL")
    duration: Optional[int] = Field(default=None, description="Video duration in seconds")
    playlist_index: int = Field(..., description="Original 1-based index in the playlist")
    playlist_title: str = Field(default="Complete C++ Placement DSA Course", description="Title of the source playlist")
    channel: str = Field(default="CodeHelp - by Babbar", description="Channel name")
    upload_date: Optional[str] = Field(default=None, description="Video upload date YYYYMMDD")
    lecture_number: Optional[int] = Field(default=None, description="Extracted lecture number if available")


class PipelineManifestEntry(BaseModel):
    video_id: str
    title: str
    playlist_index: int
    lecture_number: Optional[int] = None
    downloaded: bool = False
    transcribed: bool = False
    chunked: bool = False
    embedded: bool = False
    uploaded_to_qdrant: bool = False
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    chunks_path: Optional[str] = None
    last_updated: Optional[str] = None
    error: Optional[str] = None


class WhisperSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class VideoTranscript(BaseModel):
    video_id: str
    lecture_number: Optional[int] = None
    video_title: Optional[str] = None
    total_duration: Optional[float] = None
    segments: List[WhisperSegment] = []


class TranscriptChunk(BaseModel):
    chunk_id: str
    video_id: str
    video_title: str
    lecture_number: Optional[int] = None
    playlist_index: int
    video_url: str
    start_time: float
    end_time: float
    start_timestamp_str: str
    end_timestamp_str: str
    youtube_timestamp_url: str
    text: str
    segment_span: Optional[List[int]] = None


class TopicSource(BaseModel):
    lecture_number: Optional[int] = None
    title: str
    video_id: str
    start_time: float
    end_time: float
    timestamp_range: str
    youtube_url: str
    relevance_score: float
    matched_transcript_snippet: str
    adjacent_context: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.25
    expand_adjacent: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: List[TopicSource] = []
