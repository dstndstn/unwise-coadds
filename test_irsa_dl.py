from unwise_utils import download_frameset_1band
import sys

def main():
    frame_nums = [140, 141, 116, 139, 140, 141, 115, 116, 138, 139, 140]
    scan_ids = ['05229b', '05229b', '05232a', '05233b', '05233b', '05233b'
                '05236a', '05236a', '05237b', '05237b', '05237b']

    band = 1 # arbitrary

    frames = zip(scan_ids, frame_nums)
    for frame in frames:
        download_frameset_1band(frame[0], frame[1], band)

if __name__ == '__main__':
    sys.exit(main())
