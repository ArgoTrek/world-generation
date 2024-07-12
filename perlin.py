import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from perlin_noise import PerlinNoise

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Frontend origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the tiles and their corresponding colors
tile_colors = {
    '1': (0, 0, 1),   # Water
    '2': (1, 1, 0),   # Sand
    '3': (0, 1, 0),   # Grass
    '4': (0, 0, 1),   # Water
    '5': (0.55, 0.27, 0.07),  # Dirt
    '6': (0, 0.5, 0), # Forest
    '7': (0, 0, 1),   # Water
    '8': (0.5, 0.5, 0.5),  # Rock
    '9': (1, 1, 1)    # Snow
}

# Function to generate a Perlin noise map and map it to tiles
def generate_perlin_map(width, height, seed, base_octave=1, detail_octave=4):
    base_noise = PerlinNoise(octaves=base_octave, seed=seed)
    detail_noise = PerlinNoise(octaves=detail_octave, seed=seed)
    
    noise_map = np.zeros((width, height))
    
    for i in range(width):
        for j in range(height):
            base_value = base_noise([i/width, j/height])
            detail_value = detail_noise([i/width, j/height])
            noise_map[i][j] = base_value + 0.5 * detail_value  # Combine base and detail noise
    
    tile_map = np.zeros((width, height), dtype=str)
    
    for i in range(width):
        for j in range(height):
            value = noise_map[i][j]
            if value < -0.1:
                tile_map[i][j] = '1'  # Water (Tropics)
            elif value < 0:
                tile_map[i][j] = '2'  # Sand (Tropics)
            elif value < 0.15:
                tile_map[i][j] = '3'  # Grass (Tropics)
            elif value < 0.3:
                tile_map[i][j] = '4'  # Water (Temperate)
            elif value < 0.45:
                tile_map[i][j] = '5'  # Dirt (Temperate)
            elif value < 0.6:
                tile_map[i][j] = '6'  # Forest (Temperate)
            elif value < 0.75:
                tile_map[i][j] = '7'  # Water (Tundra)
            elif value < 0.85:
                tile_map[i][j] = '8'  # Rock (Tundra)
            else:
                tile_map[i][j] = '9'  # Snow (Tundra)
    
    return tile_map.tolist()

class WorldRequest(BaseModel):
    width: int
    height: int
    seed: int
    base_octave: int = 1
    detail_octave: int = 4

@app.post("/world/generate")
def generate_world(request: WorldRequest):
    tile_map = generate_perlin_map(request.width, request.height, request.seed, request.base_octave, request.detail_octave)
    return {"tile_map": tile_map}
