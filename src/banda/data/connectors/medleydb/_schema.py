from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class CanonicalMedleyDBRawTrack(BaseModel):
    """Represents a raw track in MedleyDB.

    Attributes:
        filename (str): Filename of the raw track.
        instrument (str): Instrument of the raw track.
    """

    filename: str = Field(..., description="Filename of the raw track.")
    instrument: str = Field(..., description="Instrument of the raw track.")


class CanonicalMedleyDBStem(BaseModel):
    """Represents a stem in MedleyDB.

    Attributes:
        component (str): Component of the stem.
        filename (str): Filename of the stem.
        instrument (str): Instrument of the stem.
        raw (Dict[str, CanonicalMedleyDBRawTrack]): Raw tracks associated with the stem.
    """

    component: str = Field(..., description="Component of the stem.")
    filename: str = Field(..., description="Filename of the stem.")
    instrument: str = Field(..., description="Instrument of the stem.")
    raw: Dict[str, CanonicalMedleyDBRawTrack] = Field(
        ..., description="Raw tracks associated with the stem."
    )


class CanonicalMedleyDBSongMetadata(BaseModel):
    """Represents the complete metadata for a MedleyDB song.

    Attributes:
        album (str): Album name.
        artist (str): Artist name.
        composer (List[str]): List of composers.
        excerpt (str): Excerpt information.
        genre (str): Genre of the song.
        has_bleed (str): Indicates if the song has bleed.
        instrumental (str): Indicates if the song is instrumental.
        mix_filename (str): Filename of the mix.
        origin (str): Origin of the song.
        producer (List[str]): List of producers.
        raw_dir (str): Raw directory.
        stem_dir (str): Stem directory.
        stems (Dict[str, CanonicalMedleyDBStem]): Stems information.
        title (str): Title of the song.
        version (float): Version of the metadata.
        website (Optional[List[str]]): List of websites.
    """

    album: str = Field(..., description="Album name.")
    artist: str = Field(..., description="Artist name.")
    composer: List[str] = Field(..., description="List of composers.")
    excerpt: str = Field(..., description="Excerpt information.")
    genre: str = Field(..., description="Genre of the song.")
    has_bleed: str = Field(..., description="Indicates if the song has bleed.")
    instrumental: str = Field(..., description="Indicates if the song is instrumental.")
    mix_filename: str = Field(..., description="Filename of the mix.")
    origin: str = Field(..., description="Origin of the song.")
    producer: List[str] = Field(..., description="List of producers.")
    raw_dir: str = Field(..., description="Raw directory.")
    stem_dir: str = Field(..., description="Stem directory.")
    stems: Dict[str, CanonicalMedleyDBStem] = Field(
        ..., description="Stems information."
    )
    title: str = Field(..., description="Title of the song.")
    version: float = Field(..., description="Version of the metadata.")
    website: Optional[List[str]] = Field(..., description="List of websites.")
