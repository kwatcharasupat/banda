from pydantic import BaseModel, Field
from typing import List, Optional
import uuid


class CanonicalMoisesDBTrack(BaseModel):
    """
    Represents a single track in a stem.
    """

    trackType: str = Field(
        ..., description="Type of the track (e.g., bass_guitar, drum_machine)."
    )
    id: uuid.UUID = Field(..., description="Unique identifier for the track.")
    type: str = Field(..., description="MIME type of the track (e.g., audio/x-wav).")
    extension: str = Field(..., description="File extension of the track (e.g., wav).")
    has_bleed: Optional[bool] = Field(
        ..., description="Indicates if the track has bleed."
    )


class CanonicalMoisesDBStem(BaseModel):
    """Represents a stem containing multiple tracks.

    Attributes:
        id (uuid.UUID): Unique identifier for the stem.
        stemName (str): Name of the stem (e.g., bass, guitar).
        tracks (List[CanonicalMoisesDBTrack]): List of tracks in the stem.
    """

    id: uuid.UUID = Field(..., description="Unique identifier for the stem.")
    stemName: str = Field(..., description="Name of the stem (e.g., bass, guitar).")
    tracks: List[CanonicalMoisesDBTrack] = Field(
        ..., description="List of tracks in the stem."
    )


class CanonicalMoisesDBSongMetadata(BaseModel):
    """Represents metadata for a song.

    Attributes:
        song (str): Name of the song.
        artist (str): Name of the artist.
        genre (str): Genre of the song.
        stems (List[CanonicalMoisesDBStem]): List of stems in the song.
    """

    song: str = Field(..., description="Name of the song.")
    artist: str = Field(..., description="Name of the artist.")
    genre: str = Field(..., description="Genre of the song.")
    stems: List[CanonicalMoisesDBStem] = Field(
        ..., description="List of stems in the song."
    )
