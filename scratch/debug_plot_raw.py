import json

def main():
    try:
        with open('storage/debug_preprocess_result.json', 'r', encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError:
        print("JSON file not found. Run preprocess first.")
        return

    # Find the segment for "あ"
    segments = [s for s in d['segments'] if s['label'] == 'あ']
    if not segments:
        print("Segment for 'あ' not found.")
        return
    
    seg = segments[0]
    traj = seg['trajectory']
    print(f"Total points in trajectory: {len(traj)}")
    
    # Filter points where pen is down
    down_pts = [p for p in traj if p['pen_state'] == 'down']
    print(f"Pen-down points: {len(down_pts)}")
    
    if not down_pts:
        print("No pen-down points found.")
        return

    xs = [p['x'] for p in down_pts]
    ys = [p['y'] for p in down_pts]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)
    
    print(f"X span: {min_x} to {max_x} (width={w})")
    print(f"Y span: {min_y} to {max_y} (height={h})")
    
    # Plot using a 40x40 grid
    grid_size = 40
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    for p in traj:
        # We can also plot pen-up states with another character if we want,
        # but let's stick to pen-down for shapes.
        if p['pen_state'] == 'down':
            gx = int((p['x'] - min_x) / w * (grid_size - 1))
            gy = int((p['y'] - min_y) / h * (grid_size - 1))
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                grid[gy][gx] = '*'
                
    print('Raw Pixel Coordinate Plot of preprocessed "あ":')
    for r in grid:
        print(''.join(r))

if __name__ == '__main__':
    main()
