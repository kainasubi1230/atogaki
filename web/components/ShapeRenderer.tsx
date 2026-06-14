export function ShapeRenderer({ type, color }: { type: string; color: string }) {
  if (type === 'square') return <div style={{ width: '100%', height: '100%', backgroundColor: color }} />;
  if (type === 'circle') return <div style={{ width: '100%', height: '100%', backgroundColor: color, borderRadius: '50%' }} />;
  if (type === 'triangle') return (
    <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polygon points="50,0 100,100 0,100" fill={color} />
    </svg>
  );
  if (type === 'arrow') return (
    <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polygon points="0,35 60,35 60,15 100,50 60,85 60,65 0,65" fill={color} />
    </svg>
  );
  return null;
}
