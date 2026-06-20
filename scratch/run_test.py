import sys, json, traceback
sys.path.append('c:/Users/s65234kh/Desktop/project/ryu-nen-kaihi-machine')
try:
    from trainerlib.preprocess import preprocess_scan
    from PIL import Image
    import io
    # create blank white image 200x200
    img = Image.new('L', (200, 200), color=255)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    result = preprocess_scan(image_bytes)
    print('Success?', result.get('success'))
    print('Segment count', result.get('segment_count'))
    print('Writer style keys', list(result.get('writer_style', {}).keys()))
except Exception as e:
    traceback.print_exc()
