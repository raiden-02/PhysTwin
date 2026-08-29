const OPENCV_TO_THREE = [
  1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1,
];

export function multiply4(a: number[], b: number[]): number[] {
  const out = new Array<number>(16);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      out[row * 4 + column] =
        a[row * 4] * b[column] +
        a[row * 4 + 1] * b[4 + column] +
        a[row * 4 + 2] * b[8 + column] +
        a[row * 4 + 3] * b[12 + column];
    }
  }
  return out;
}

export function opencvCameraToThreeWorld(T_world_camera: number[]): number[] {
  return multiply4(T_world_camera, OPENCV_TO_THREE);
}

export function nearestSampleIndex(timestamps: number[], time: number): number {
  if (timestamps.length === 0) return 0;
  let best = 0;
  let bestDist = Math.abs(timestamps[0] - time);
  for (let index = 1; index < timestamps.length; index += 1) {
    const dist = Math.abs(timestamps[index] - time);
    if (dist < bestDist) {
      best = index;
      bestDist = dist;
    }
  }
  return best;
}

export function verticalFovDeg(fyPx: number, heightPx: number): number {
  return (2 * Math.atan(heightPx / (2 * fyPx)) * 180) / Math.PI;
}
