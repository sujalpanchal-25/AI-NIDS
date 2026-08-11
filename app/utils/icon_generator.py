"""
Pure Python PNG Icon Generator for AI-NIDS PWA Assets
=====================================================
Generates valid PNG icons without external dependencies (using zlib & struct).
"""

import os
import zlib
import struct

def generate_png(width: int, height: int, r: int = 59, g: int = 130, b: int = 246, a: int = 255) -> bytes:
    """Generate valid PNG binary data for given width, height and color."""
    # PNG Signature
    png_sig = b'\x89PNG\r\n\x1a\n'
    
    # IHDR Chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
    ihdr_chunk = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    # IDAT Chunk (Raw RGBA scanlines)
    raw_row = b'\x00' + bytes([r, g, b, a]) * width
    raw_data = raw_row * height
    compressed_data = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b'IDAT' + compressed_data)
    idat_chunk = struct.pack('>I', len(compressed_data)) + b'IDAT' + compressed_data + struct.pack('>I', idat_crc)
    
    # IEND Chunk
    iend_crc = zlib.crc32(b'IEND')
    iend_chunk = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    return png_sig + ihdr_chunk + idat_chunk + iend_chunk


def ensure_pwa_icons(icons_dir: str):
    """Ensure all required PWA icons exist in icons_dir."""
    os.makedirs(icons_dir, exist_ok=True)
    sizes = [72, 96, 128, 144, 152, 192, 384, 512]
    
    for size in sizes:
        icon_path = os.path.join(icons_dir, f"icon-{size}x{size}.png")
        if not os.path.exists(icon_path):
            png_bytes = generate_png(size, size, r=59, g=130, b=246, a=255)
            with open(icon_path, 'wb') as f:
                f.write(png_bytes)


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    icons_path = os.path.join(project_root, 'app', 'static', 'images', 'icons')
    ensure_pwa_icons(icons_path)
