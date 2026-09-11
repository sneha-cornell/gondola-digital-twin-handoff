import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GizmoHelper, GizmoViewport, Html, Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";

const SHELF_POINT_PALETTE = ["#8fd8ff", "#ffd386", "#9fe4be", "#d4bbff"];

function makeBasis(widthAxis, heightAxis, depthAxis) {
  const basis = new THREE.Matrix4();
  basis.makeBasis(widthAxis, heightAxis, depthAxis);
  return new THREE.Quaternion().setFromRotationMatrix(basis);
}

function formatDimensionLength(value) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  if (value < 0.5) {
    return `${Math.round(value * 100)} cm`;
  }
  return `${value.toFixed(2)} m`;
}

function pointFromBasis(widthAxis, heightAxis, depthAxis, widthValue, heightValue, depthValue) {
  return widthAxis
    .clone()
    .multiplyScalar(widthValue)
    .add(heightAxis.clone().multiplyScalar(heightValue))
    .add(depthAxis.clone().multiplyScalar(depthValue));
}

function snapValue(value, step) {
  return Math.round(value / step) * step;
}

function clampToBounds(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function productHasReliableName(product) {
  return Boolean(product?.product_identity_label && (product?.display_label ?? product?.label));
}

function getMarkerTone(product, active) {
  if (active) {
    return {
      body: "#f59e0b",
      glow: "#fde68a",
      rim: "#7c2d12",
      ray: "#171717"
    };
  }
  if (productHasReliableName(product)) {
    return {
      body: "#eef2f7",
      glow: "#f8fafc",
      rim: "#505866",
      ray: "#202020"
    };
  }
  return {
    body: "#d3d9e1",
    glow: "#eef2f7",
    rim: "#4b5563",
    ray: "#262626"
  };
}

function worldPointFromShelfCoordinates(shelf, widthValue, heightValue, depthValue) {
  const widthAxis = new THREE.Vector3(...shelf.axes[shelf.geometry.width_axis]).normalize();
  const depthAxis = new THREE.Vector3(...shelf.axes[shelf.geometry.depth_axis]).normalize();
  const normalAxis = new THREE.Vector3(...shelf.normal).normalize();
  return pointFromBasis(widthAxis, normalAxis, depthAxis, widthValue, heightValue, depthValue);
}

function projectWorldPointToShelfTop(shelf, point, lift = 0.004) {
  if (!shelf?.geometry || !point) {
    return null;
  }

  const worldPoint = Array.isArray(point) ? new THREE.Vector3(...point) : point.clone();
  const widthAxis = new THREE.Vector3(...shelf.axes[shelf.geometry.width_axis]).normalize();
  const depthAxis = new THREE.Vector3(...shelf.axes[shelf.geometry.depth_axis]).normalize();
  const widthBounds = shelf.geometry.width_bounds;
  const depthBounds = shelf.geometry.depth_bounds;
  const widthValue = clampToBounds(worldPoint.dot(widthAxis), widthBounds[0], widthBounds[1]);
  const depthValue = clampToBounds(worldPoint.dot(depthAxis), depthBounds[0], depthBounds[1]);

  return worldPointFromShelfCoordinates(shelf, widthValue, shelf.geometry.height_bounds[1] + lift, depthValue);
}

function projectPointerToAxis(clientX, clientY, linePoint, lineDirection, camera, domElement) {
  const bounds = domElement.getBoundingClientRect();
  if (bounds.width === 0 || bounds.height === 0) {
    return null;
  }

  const pointer = new THREE.Vector2(
    ((clientX - bounds.left) / bounds.width) * 2 - 1,
    -((clientY - bounds.top) / bounds.height) * 2 + 1
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const rayOrigin = raycaster.ray.origin.clone();
  const rayDirection = raycaster.ray.direction.clone().normalize();
  const axisDirection = lineDirection.clone().normalize();
  const delta = linePoint.clone().sub(rayOrigin);
  const axisDot = axisDirection.dot(rayDirection);
  const denominator = 1 - axisDot * axisDot;

  if (Math.abs(denominator) < 1e-5) {
    return null;
  }

  const scalar = (axisDot * rayDirection.dot(delta) - axisDirection.dot(delta)) / denominator;
  return linePoint.clone().add(axisDirection.clone().multiplyScalar(scalar));
}

function makePartStyle(kind, xray) {
  const opacityScale = xray ? 0.3 : 0.72;
  switch (kind) {
    case "side_panel":
      return { color: "#d7d9de", opacity: 0.9 * opacityScale, roughness: 0.5, metalness: 0.04 };
    case "back_panel":
      return { color: "#bfc4cb", opacity: 0.84 * opacityScale, roughness: 0.62, metalness: 0.02 };
    case "plinth":
    case "kickplate":
      return { color: "#a4abb5", opacity: 0.88 * opacityScale, roughness: 0.6, metalness: 0.08 };
    case "top_cap":
      return { color: "#d9dde3", opacity: 0.86 * opacityScale, roughness: 0.48, metalness: 0.04 };
    default:
      return { color: "#b4bac3", opacity: 0.82 * opacityScale, roughness: 0.56, metalness: 0.06 };
  }
}

function GeometryPart({ part, clippingPlanes, xray }) {
  const widthAxis = useMemo(() => new THREE.Vector3(...part.axes.width), [part.axes.width]);
  const heightAxis = useMemo(() => new THREE.Vector3(...part.axes.height), [part.axes.height]);
  const depthAxis = useMemo(() => new THREE.Vector3(...part.axes.depth), [part.axes.depth]);
  const center = useMemo(() => new THREE.Vector3(...part.center), [part.center]);
  const style = useMemo(() => makePartStyle(part.kind, xray), [part.kind, xray]);
  const quaternion = useMemo(
    () => makeBasis(widthAxis, heightAxis, depthAxis),
    [widthAxis, heightAxis, depthAxis]
  );

  return (
    <mesh position={center} quaternion={quaternion} castShadow receiveShadow>
      <boxGeometry args={part.size} />
      <meshStandardMaterial
        color={part.style?.color ?? style.color}
        transparent
        opacity={Math.min((part.style?.opacity ?? 1) * style.opacity, 0.74)}
        roughness={style.roughness}
        metalness={style.metalness}
        clippingPlanes={clippingPlanes}
      />
    </mesh>
  );
}

function RackGeometry({ rack, clippingPlanes, xray }) {
  if (!rack?.parts?.length) {
    return null;
  }

  return (
    <group>
      {rack.parts.map((part) => (
        <GeometryPart key={part.id} part={part} clippingPlanes={clippingPlanes} xray={xray} />
      ))}
    </group>
  );
}

function ShelfMesh({ shelf, highlighted, wireframe, xray, clippingPlanes, onSelect }) {
  const geometry = useMemo(() => {
    const meshGeometry = new THREE.BufferGeometry();
    meshGeometry.setAttribute("position", new THREE.Float32BufferAttribute(shelf.mesh.vertices.flat(), 3));
    meshGeometry.setIndex(shelf.mesh.triangles.flat());
    meshGeometry.computeVertexNormals();
    return meshGeometry;
  }, [shelf]);
  const edgeGeometry = useMemo(() => new THREE.EdgesGeometry(geometry, 15), [geometry]);

  return (
    <group>
      <mesh geometry={geometry} onClick={() => onSelect(shelf.id)} castShadow receiveShadow>
        <meshPhysicalMaterial
          color={highlighted ? "#d8dbe0" : "#cfd3d8"}
          emissive={highlighted ? "#6b7280" : "#42474f"}
          emissiveIntensity={highlighted ? 0.12 : 0.02}
          side={THREE.DoubleSide}
          roughness={0.42}
          metalness={0.02}
          clearcoat={0.08}
          clearcoatRoughness={0.8}
          transparent={xray}
          opacity={xray ? 0.24 : 0.98}
          wireframe={wireframe}
          clippingPlanes={clippingPlanes}
        />
      </mesh>
      {!wireframe && (
        <lineSegments geometry={edgeGeometry}>
          <lineBasicMaterial color={highlighted ? "#555b64" : "#444b55"} transparent opacity={0.9} />
        </lineSegments>
      )}
    </group>
  );
}

function ShelfFrontLip({ shelf, highlighted, clippingPlanes, xray }) {
  const widthAxisName = shelf.geometry.width_axis;
  const depthAxisName = shelf.geometry.depth_axis;
  const widthAxis = useMemo(() => new THREE.Vector3(...shelf.axes[widthAxisName]), [shelf, widthAxisName]);
  const depthAxis = useMemo(() => new THREE.Vector3(...shelf.axes[depthAxisName]), [shelf, depthAxisName]);
  const normalAxis = useMemo(() => new THREE.Vector3(...shelf.normal), [shelf]);
  const quaternion = useMemo(() => makeBasis(widthAxis, normalAxis, depthAxis), [widthAxis, normalAxis, depthAxis]);
  const widthBounds = shelf.geometry.width_bounds;
  const depthBounds = shelf.geometry.depth_bounds;
  const heightBounds = shelf.geometry.height_bounds;
  const widthSpan = widthBounds[1] - widthBounds[0];
  const depthSpan = depthBounds[1] - depthBounds[0];
  const thickness = heightBounds[1] - heightBounds[0];
  const lipDepth = Math.max(0.02, depthSpan * 0.045);
  const lipHeight = Math.max(0.05, thickness * 2.2);
  const centerWidth = (widthBounds[0] + widthBounds[1]) * 0.5;
  const centerDepth = depthBounds[1] + lipDepth * 0.4;
  const centerHeight = heightBounds[1] + lipHeight * 0.4;
  const center = useMemo(
    () =>
      widthAxis
        .clone()
        .multiplyScalar(centerWidth)
        .add(normalAxis.clone().multiplyScalar(centerHeight))
        .add(depthAxis.clone().multiplyScalar(centerDepth)),
    [centerDepth, centerHeight, centerWidth, depthAxis, normalAxis, widthAxis]
  );

  return (
    <mesh position={center} quaternion={quaternion} castShadow receiveShadow>
      <boxGeometry args={[widthSpan, lipHeight, lipDepth]} />
      <meshStandardMaterial
        color={highlighted ? "#d3d7dd" : "#c4c9cf"}
        emissive={highlighted ? "#6b7280" : "#2d3440"}
        emissiveIntensity={highlighted ? 0.08 : 0.03}
        roughness={0.4}
        metalness={0.04}
        transparent={xray}
        opacity={xray ? 0.28 : 0.98}
        clippingPlanes={clippingPlanes}
      />
    </mesh>
  );
}

function ShelfClusterPoints({ shelf, shelfIndex, highlighted, dimmed, clippingPlanes }) {
  const geometry = useMemo(() => {
    const sourcePoints = shelf?._cluster_points ?? [];
    if (!sourcePoints.length) {
      return null;
    }
    const stride = sourcePoints.length > 14000 ? Math.ceil(sourcePoints.length / 14000) : 1;
    const positions = [];
    for (let index = 0; index < sourcePoints.length; index += stride) {
      positions.push(...sourcePoints[index]);
    }
    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return pointsGeometry;
  }, [shelf]);

  if (!geometry) {
    return null;
  }

  const color = highlighted ? "#ffd166" : SHELF_POINT_PALETTE[shelfIndex % SHELF_POINT_PALETTE.length];
  const opacity = dimmed ? 0.18 : highlighted ? 0.92 : 0.58;

  return (
    <points geometry={geometry} renderOrder={1}>
      <pointsMaterial
        color={color}
        size={highlighted ? 0.034 : 0.024}
        sizeAttenuation
        transparent
        opacity={opacity}
        depthTest={false}
        depthWrite={false}
        clippingPlanes={clippingPlanes}
      />
    </points>
  );
}

function DashedDebugRay({ start, end, color, active = false }) {
  const lineRef = useRef(null);
  const geometry = useMemo(() => {
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    return lineGeometry;
  }, [end, start]);

  useEffect(() => {
    lineRef.current?.computeLineDistances();
    return () => {
      geometry.dispose();
    };
  }, [geometry]);

  return (
    <line ref={lineRef} geometry={geometry} renderOrder={4}>
      <lineDashedMaterial
        color={color}
        dashSize={active ? 0.08 : 0.055}
        gapSize={active ? 0.042 : 0.032}
        transparent
        opacity={active ? 0.9 : 0.58}
        depthTest={false}
      />
    </line>
  );
}

function ProductMarker({ product, shelf, active, rayOrigin, showRay, onSelect, clippingPlanes }) {
  if (!shelf?.geometry) {
    return null;
  }

  const normalAxis = useMemo(() => new THREE.Vector3(...shelf.normal).normalize(), [shelf]);
  const markerRadius = clampToBounds((shelf.extents?.depth ?? 1) * 0.02, 0.048, 0.075);
  const anchorPoint = useMemo(() => new THREE.Vector3(...product.p3d), [product.p3d]);
  const surfacePoint = useMemo(() => {
    if (Array.isArray(product.projection_point)) {
      return new THREE.Vector3(...product.projection_point);
    }
    return projectWorldPointToShelfTop(shelf, anchorPoint, 0.003) ?? anchorPoint.clone();
  }, [anchorPoint, product.projection_point, shelf]);
  const markerPoint = useMemo(
    () => surfacePoint.clone().add(normalAxis.clone().multiplyScalar(markerRadius * 0.34)),
    [markerRadius, normalAxis, surfacePoint]
  );
  const tone = useMemo(() => getMarkerTone(product, active), [active, product]);
  const label = product.display_label ?? product.label ?? product.id;

  return (
    <group
      position={markerPoint}
      onClick={(event) => {
        event.stopPropagation();
        onSelect(product);
      }}
      onPointerOver={(event) => {
        event.stopPropagation();
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        document.body.style.cursor = "";
      }}
      renderOrder={10}
    >
      {showRay && rayOrigin ? (
        <DashedDebugRay start={rayOrigin} end={markerPoint} color={tone.ray} active={active} />
      ) : null}
      <mesh position={surfacePoint.clone().sub(markerPoint)}>
        <cylinderGeometry args={[markerRadius * 0.44, markerRadius * 0.44, markerRadius * 0.16, 24]} />
        <meshStandardMaterial
          color="#979da7"
          emissive="#404651"
          emissiveIntensity={0.08}
          roughness={0.52}
          metalness={0.02}
          depthTest={false}
          depthWrite={false}
          clippingPlanes={clippingPlanes}
        />
      </mesh>
      {active ? (
        <mesh scale={1.6}>
          <sphereGeometry args={[markerRadius, 24, 24]} />
          <meshStandardMaterial
            color={tone.glow}
            emissive={tone.glow}
            emissiveIntensity={0.16}
            transparent
            opacity={0.22}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
      ) : null}
      <mesh scale={1.12}>
        <sphereGeometry args={[markerRadius, 20, 20]} />
        <meshStandardMaterial
          color={tone.rim}
          emissive={tone.rim}
          emissiveIntensity={0.03}
          transparent
          opacity={0.96}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>
      <mesh castShadow receiveShadow>
        <sphereGeometry args={[markerRadius, 24, 24]} />
        <meshStandardMaterial
          color={tone.body}
          emissive={active ? tone.glow : "#ffffff"}
          emissiveIntensity={active ? 0.18 : 0.03}
          roughness={0.42}
          metalness={0.08}
          depthTest={false}
          depthWrite={false}
          clippingPlanes={clippingPlanes}
        />
      </mesh>
      {active ? (
        <Html position={[0, markerRadius * 2.4, 0]} center transform sprite distanceFactor={8}>
          <div
            style={{
              padding: "4px 8px",
              borderRadius: "999px",
              background: "rgba(24, 24, 27, 0.94)",
              border: "1px solid rgba(245, 158, 11, 0.3)",
              color: "#f8fafc",
              fontSize: "10px",
              fontWeight: 700,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              whiteSpace: "nowrap",
              boxShadow: "0 10px 24px rgba(5, 10, 18, 0.3)"
            }}
          >
            {label}
          </div>
        </Html>
      ) : null}
    </group>
  );
}

function SceneInstruction({ position, productCount }) {
  return (
    <Html position={position} center transform sprite distanceFactor={10}>
      <div
            style={{
              padding: "6px 10px",
              borderRadius: "999px",
              background: "rgba(78, 83, 90, 0.72)",
              border: "1px solid rgba(255, 255, 255, 0.18)",
              color: "#ebedf0",
          fontSize: "11px",
          fontWeight: 600,
          letterSpacing: "0.02em",
          whiteSpace: "nowrap",
          boxShadow: "0 8px 20px rgba(20, 22, 26, 0.16)"
        }}
      >
        {productCount} perspective-mapped detections. Select a marker to inspect it.
      </div>
    </Html>
  );
}

function DimensionOverlay({
  start,
  end,
  offsetVector,
  tickDirection,
  label,
  color,
  labelBackground,
  labelColor
}) {
  const dimensionStart = useMemo(() => start.clone().add(offsetVector), [offsetVector, start]);
  const dimensionEnd = useMemo(() => end.clone().add(offsetVector), [end, offsetVector]);
  const lineDirection = useMemo(() => end.clone().sub(start).normalize(), [end, start]);
  const tickVector = useMemo(() => {
    const candidate = tickDirection.clone().normalize();
    if (Math.abs(candidate.dot(lineDirection)) > 0.94) {
      return new THREE.Vector3(0, 1, 0);
    }
    return candidate;
  }, [lineDirection, tickDirection]);
  const tickSize = useMemo(() => {
    const lineLength = start.distanceTo(end);
    return Math.max(Math.min(lineLength * 0.08, 0.22), 0.06);
  }, [end, start]);
  const labelPosition = useMemo(
    () =>
      dimensionStart
        .clone()
        .lerp(dimensionEnd, 0.5)
        .add(tickVector.clone().multiplyScalar(tickSize * 0.9)),
    [dimensionEnd, dimensionStart, tickSize, tickVector]
  );
  const startTick = useMemo(
    () => [
      dimensionStart.clone().add(tickVector.clone().multiplyScalar(-tickSize * 0.5)),
      dimensionStart.clone().add(tickVector.clone().multiplyScalar(tickSize * 0.5))
    ],
    [dimensionStart, tickSize, tickVector]
  );
  const endTick = useMemo(
    () => [
      dimensionEnd.clone().add(tickVector.clone().multiplyScalar(-tickSize * 0.5)),
      dimensionEnd.clone().add(tickVector.clone().multiplyScalar(tickSize * 0.5))
    ],
    [dimensionEnd, tickSize, tickVector]
  );
  const extensionMaterialProps = { color, opacity: 0.72, transparent: true, depthTest: false };
  const mainMaterialProps = { color, opacity: 0.94, transparent: true, depthTest: false };

  return (
    <group renderOrder={12}>
      <Line points={[start, dimensionStart]} lineWidth={0.9} {...extensionMaterialProps} />
      <Line points={[end, dimensionEnd]} lineWidth={0.9} {...extensionMaterialProps} />
      <Line points={[dimensionStart, dimensionEnd]} lineWidth={1.2} {...mainMaterialProps} />
      <Line points={startTick} lineWidth={1.2} {...mainMaterialProps} />
      <Line points={endTick} lineWidth={1.2} {...mainMaterialProps} />
      <Html position={labelPosition} transform sprite distanceFactor={8} occlude={false}>
        <div
          style={{
            padding: "4px 8px",
            borderRadius: "999px",
            border: `1px solid ${color}55`,
            background: labelBackground,
            color: labelColor,
            fontSize: "11px",
            fontWeight: 700,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            whiteSpace: "nowrap",
            boxShadow: "0 10px 28px rgba(2, 6, 23, 0.3)"
          }}
        >
          {label}
        </div>
      </Html>
    </group>
  );
}

function buildRackDimensionSet(rack, widthSpanOverride) {
  if (!rack?.axes || !rack?.bounds) {
    return null;
  }

  const widthAxis = new THREE.Vector3(...rack.axes.width).normalize();
  const heightAxis = new THREE.Vector3(...rack.axes.height).normalize();
  const depthAxis = new THREE.Vector3(...rack.axes.depth).normalize();
  const [rawWidthMin, rawWidthMax] = rack.bounds.width;
  const [heightMin, heightMax] = rack.bounds.height;
  const [depthMin, depthMax] = rack.bounds.depth;
  const widthCenter = (rawWidthMin + rawWidthMax) * 0.5;
  const widthSpan = widthSpanOverride ?? rawWidthMax - rawWidthMin;
  const widthMin = widthCenter - widthSpan * 0.5;
  const widthMax = widthCenter + widthSpan * 0.5;
  const heightSpan = heightMax - heightMin;
  const depthSpan = depthMax - depthMin;
  const widthLift = heightSpan * 0.12 + 0.14;
  const depthLift = widthSpan * 0.07 + 0.12;
  const heightLift = widthSpan * 0.1 + 0.14;

  const widthOffsetVector = depthAxis.clone().multiplyScalar(depthSpan * 0.18 + 0.16).add(heightAxis.clone().multiplyScalar(-widthLift));
  const widthStart = pointFromBasis(widthAxis, heightAxis, depthAxis, widthMin, heightMin, depthMax);
  const widthEnd = pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMin, depthMax);
  const widthMid = widthStart.clone().add(widthOffsetVector).lerp(widthEnd.clone().add(widthOffsetVector), 0.5);

  return {
    dimensions: [
      {
        id: "rack-width",
        start: widthStart,
        end: widthEnd,
        offsetVector: widthOffsetVector,
        tickDirection: heightAxis,
        label: `W ${formatDimensionLength(widthSpan)}`
      },
      {
        id: "rack-depth",
        start: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMin, depthMin),
        end: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMin, depthMax),
        offsetVector: widthAxis.clone().multiplyScalar(-depthLift).add(heightAxis.clone().multiplyScalar(-heightSpan * 0.06 - 0.08)),
        tickDirection: widthAxis,
        label: `D ${formatDimensionLength(depthSpan)}`
      },
      {
        id: "rack-height",
        start: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMin, heightMin, depthMin),
        end: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMin, heightMax, depthMin),
        offsetVector: widthAxis.clone().multiplyScalar(heightLift).add(depthAxis.clone().multiplyScalar(-depthSpan * 0.14 - 0.12)),
        tickDirection: depthAxis,
        label: `H ${formatDimensionLength(heightSpan)}`
      }
    ],
    handles: {
      width: {
        axis: widthAxis,
        linePoint: widthMid,
        minPosition: widthStart.clone().add(widthOffsetVector),
        maxPosition: widthEnd.clone().add(widthOffsetVector)
      }
    }
  };
}

function buildShelfDimensionSet(shelf, offsetDelta = 0) {
  if (!shelf?.geometry) {
    return null;
  }

  const widthAxis = new THREE.Vector3(...shelf.axes[shelf.geometry.width_axis]).normalize();
  const depthAxis = new THREE.Vector3(...shelf.axes[shelf.geometry.depth_axis]).normalize();
  const heightAxis = new THREE.Vector3(...shelf.normal).normalize();
  const [widthMin, widthMax] = shelf.geometry.width_bounds;
  const [depthMin, depthMax] = shelf.geometry.depth_bounds;
  const [heightMin, heightMax] = shelf.geometry.height_bounds;
  const widthSpan = widthMax - widthMin;
  const depthSpan = depthMax - depthMin;
  const heightSpan = heightMax - heightMin;
  const elevation = Math.max(heightSpan * 1.8, 0.09);
  const sideShift = Math.max(widthSpan * 0.08, 0.12);
  const translation = heightAxis.clone().multiplyScalar(offsetDelta);

  const widthStart = pointFromBasis(widthAxis, heightAxis, depthAxis, widthMin, heightMax, depthMax).add(translation);
  const widthEnd = pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMax, depthMax).add(translation);
  const heightHandlePosition = pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMax, depthMax)
    .add(translation)
    .add(depthAxis.clone().multiplyScalar(depthSpan * 0.22 + 0.14))
    .add(heightAxis.clone().multiplyScalar(elevation * 0.66));

  return {
    dimensions: [
      {
        id: `${shelf.id}-width`,
        start: widthStart,
        end: widthEnd,
        offsetVector: depthAxis.clone().multiplyScalar(depthSpan * 0.15 + 0.12).add(heightAxis.clone().multiplyScalar(elevation)),
        tickDirection: heightAxis,
        label: `Shelf W ${formatDimensionLength(widthSpan)}`
      },
      {
        id: `${shelf.id}-depth`,
        start: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMax, depthMin).add(translation),
        end: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMax, depthMax).add(translation),
        offsetVector: widthAxis.clone().multiplyScalar(-sideShift).add(heightAxis.clone().multiplyScalar(elevation * 0.92)),
        tickDirection: widthAxis,
        label: `Shelf D ${formatDimensionLength(depthSpan)}`
      },
      {
        id: `${shelf.id}-thickness`,
        start: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMin, depthMax).add(translation),
        end: pointFromBasis(widthAxis, heightAxis, depthAxis, widthMax, heightMax, depthMax).add(translation),
        offsetVector: widthAxis.clone().multiplyScalar(-sideShift * 1.16).add(depthAxis.clone().multiplyScalar(depthSpan * 0.18 + 0.1)),
        tickDirection: depthAxis,
        label: `Deck ${formatDimensionLength(heightSpan)}`
      }
    ],
    handle: {
      axis: heightAxis,
      linePoint: heightHandlePosition,
      position: heightHandlePosition
    }
  };
}

function DimensionCollection({ dimensions, color, labelBackground, labelColor }) {
  if (!dimensions?.length) {
    return null;
  }

  return (
    <group>
      {dimensions.map((dimension) => (
        <DimensionOverlay
          key={dimension.id}
          {...dimension}
          color={color}
          labelBackground={labelBackground}
          labelColor={labelColor}
        />
      ))}
    </group>
  );
}

function RackDimensions({ dimensionSet }) {
  return (
    <DimensionCollection
      dimensions={dimensionSet?.dimensions ?? []}
      color="#7dd3fc"
      labelBackground="rgba(6, 19, 31, 0.9)"
      labelColor="#d9f5ff"
    />
  );
}

function ShelfDimensions({ dimensionSet }) {
  return (
    <DimensionCollection
      dimensions={dimensionSet?.dimensions ?? []}
      color="#f8c34f"
      labelBackground="rgba(42, 18, 3, 0.9)"
      labelColor="#fff2cc"
    />
  );
}

function ViewportHandle({ position, active, disabled, color, label, onPointerDown }) {
  const tint = disabled ? "#475569" : color;

  return (
    <group position={position} renderOrder={14}>
      <mesh
        onPointerDown={(event) => {
          event.stopPropagation();
          if (!disabled) {
            document.body.style.cursor = "grabbing";
            onPointerDown(event);
          }
        }}
        onPointerOver={(event) => {
          event.stopPropagation();
          if (!disabled) {
            document.body.style.cursor = "grab";
          }
        }}
        onPointerOut={() => {
          if (!active) {
            document.body.style.cursor = "";
          }
        }}
        castShadow
      >
        <sphereGeometry args={[active ? 0.06 : 0.045, 24, 24]} />
        <meshStandardMaterial
          color={tint}
          emissive={active ? tint : "#111827"}
          emissiveIntensity={active ? 0.55 : 0.12}
          roughness={0.26}
          metalness={0.18}
          transparent={disabled}
          opacity={disabled ? 0.55 : 1}
        />
      </mesh>
      <mesh rotation={[Math.PI * 0.5, 0, 0]}>
        <torusGeometry args={[0.075, 0.008, 12, 48]} />
        <meshStandardMaterial color={tint} emissive={tint} emissiveIntensity={active ? 0.24 : 0.08} />
      </mesh>
      <Html position={[0, 0.12, 0]} center transform sprite distanceFactor={8}>
        <div
          style={{
            padding: "3px 7px",
            borderRadius: "999px",
            background: "rgba(6, 12, 18, 0.84)",
            border: `1px solid ${tint}55`,
            color: "#e5edf8",
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            whiteSpace: "nowrap"
          }}
        >
          {label}
        </div>
      </Html>
    </group>
  );
}

// Triggers a re-render whenever any of the watched values change (needed with frameloop="demand").
function InvalidateOnChange({ values }) {
  const { invalidate } = useThree();
  useEffect(() => { invalidate(); }, values); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

// Draws all inactive markers as two InstancedMeshes (rim + body) — one draw call each.
// The selected marker is rendered separately by ProductMarker with its full detail.
function MarkersLayer({ products, shelves, selectedId, onSelect, clippingPlanes }) {
  const rimRef = useRef(null);
  const bodyRef = useRef(null);
  const { invalidate } = useThree();

  const shelfById = useMemo(() => new Map(shelves.map((s) => [s.id, s])), [shelves]);

  const markerData = useMemo(() => {
    const result = [];
    for (const product of products) {
      if (product.id === selectedId) continue;
      const shelf = shelfById.get(product.shelf_id);
      if (!shelf?.geometry) continue;
      const markerRadius = Math.min(Math.max((shelf.extents?.depth ?? 1) * 0.02, 0.048), 0.075);
      const anchorPoint = new THREE.Vector3(...product.p3d);
      const surfacePoint = Array.isArray(product.projection_point)
        ? new THREE.Vector3(...product.projection_point)
        : (projectWorldPointToShelfTop(shelf, anchorPoint, 0.003) ?? anchorPoint.clone());
      const normalAxis = new THREE.Vector3(...shelf.normal).normalize();
      const markerPoint = surfacePoint.clone().add(normalAxis.clone().multiplyScalar(markerRadius * 0.34));
      const tone = getMarkerTone(product, false);
      result.push({ product, markerPoint, markerRadius, tone });
    }
    return result;
  }, [products, selectedId, shelfById]);

  useEffect(() => {
    const rimMesh = rimRef.current;
    const bodyMesh = bodyRef.current;
    if (!rimMesh || !bodyMesh) return;
    const matrix = new THREE.Matrix4();
    markerData.forEach(({ markerPoint, markerRadius }, i) => {
      matrix.makeScale(markerRadius * 1.12, markerRadius * 1.12, markerRadius * 1.12);
      matrix.setPosition(markerPoint);
      rimMesh.setMatrixAt(i, matrix);
      matrix.makeScale(markerRadius, markerRadius, markerRadius);
      matrix.setPosition(markerPoint);
      bodyMesh.setMatrixAt(i, matrix);
    });
    rimMesh.instanceMatrix.needsUpdate = true;
    bodyMesh.instanceMatrix.needsUpdate = true;
    invalidate();
  }, [markerData, invalidate]);

  const count = markerData.length;
  if (count === 0) return null;

  // Click detection: find which instance was clicked by comparing instance index.
  function handleBodyClick(event) {
    event.stopPropagation();
    const idx = event.instanceId;
    if (idx != null && markerData[idx]) {
      onSelect(markerData[idx].product);
    }
  }

  return (
    <group renderOrder={10}>
      <instancedMesh ref={rimRef} args={[null, null, count]} renderOrder={10}
        onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { document.body.style.cursor = ""; }}
        onClick={handleBodyClick}
      >
        <sphereGeometry args={[1, 12, 12]} />
        <meshStandardMaterial color="#505866" transparent opacity={0.96} depthTest={false} depthWrite={false} clippingPlanes={clippingPlanes} />
      </instancedMesh>
      <instancedMesh ref={bodyRef} args={[null, null, count]} renderOrder={11}
        onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { document.body.style.cursor = ""; }}
        onClick={handleBodyClick}
      >
        <sphereGeometry args={[1, 12, 12]} />
        <meshStandardMaterial color="#d3d9e1" transparent opacity={0.96} depthTest={false} depthWrite={false} clippingPlanes={clippingPlanes} />
      </instancedMesh>
    </group>
  );
}

function EmptyScene() {
  return (
    <mesh position={[0, 0, 0]} receiveShadow>
      <boxGeometry args={[1.5, 0.08, 0.6]} />
      <meshStandardMaterial color="#475569" transparent opacity={0.35} />
    </mesh>
  );
}

function buildDebugRayOrigin(rack) {
  if (!rack?.axes || !rack?.bounds) {
    return new THREE.Vector3(0, 0, 0);
  }

  const widthAxis = new THREE.Vector3(...rack.axes.width).normalize();
  const heightAxis = new THREE.Vector3(...rack.axes.height).normalize();
  const depthAxis = new THREE.Vector3(...rack.axes.depth).normalize();
  const [widthMin, widthMax] = rack.bounds.width;
  const [heightMin, heightMax] = rack.bounds.height;
  const [depthMin, depthMax] = rack.bounds.depth;
  const widthPad = Math.max((widthMax - widthMin) * 0.05, 0.18);
  const heightPad = Math.max((heightMax - heightMin) * 0.08, 0.12);
  const depthPad = Math.max((depthMax - depthMin) * 0.14, 0.24);

  return pointFromBasis(
    widthAxis,
    heightAxis,
    depthAxis,
    widthMax + widthPad,
    heightMin - heightPad,
    depthMax + depthPad
  );
}

function buildSceneMetrics(rack, shelves, products) {
  const box = new THREE.Box3();
  const fallbackPointBox = new THREE.Box3();

  if (rack?.parts?.length) {
    for (const part of rack.parts) {
      for (const vertex of part.vertices ?? []) {
        box.expandByPoint(new THREE.Vector3(...vertex));
      }
    }
  }

  for (const shelf of shelves) {
    for (const vertex of shelf.mesh.vertices) {
      box.expandByPoint(new THREE.Vector3(...vertex));
    }
    for (const point of shelf._cluster_points ?? []) {
      fallbackPointBox.expandByPoint(new THREE.Vector3(...point));
    }
  }

  for (const product of products) {
    const point = Array.isArray(product.projection_point) ? product.projection_point : product.p3d;
    box.expandByPoint(new THREE.Vector3(...point));
  }

  if (box.isEmpty()) {
    if (!fallbackPointBox.isEmpty()) {
      box.copy(fallbackPointBox);
    } else {
      box.setFromCenterAndSize(new THREE.Vector3(0, 0, 0), new THREE.Vector3(2, 1.4, 1.4));
    }
  }

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z);
  const worldUp = new THREE.Vector3(0, 1, 0);
  const axes = rack?.axes
    ? (() => {
        const width = new THREE.Vector3(...rack.axes.width).normalize();
        const height = new THREE.Vector3(...rack.axes.height).normalize();
        const rawDepth = new THREE.Vector3(...rack.axes.depth).normalize();

        // COLMAP world axes are arbitrary. Normalize the view basis so "up" points
        // upward on screen instead of following a potentially downward scene normal.
        if (height.dot(worldUp) < 0) {
          height.multiplyScalar(-1);
        }

        const depth = new THREE.Vector3().crossVectors(width, height).normalize();
        if (depth.dot(rawDepth) < 0) {
          depth.multiplyScalar(-1);
        }

        return { width, height, depth };
      })()
    : {
        width: new THREE.Vector3(1, 0, 0),
        height: new THREE.Vector3(0, 1, 0),
        depth: new THREE.Vector3(0, 0, 1)
      };

  return { box, center, size, radius, axes };
}

function ViewPresetController({ metrics, preset, controlsRef }) {
  const { camera } = useThree();

  useEffect(() => {
    const distance = Math.max(metrics.radius * 2.2, 2.4);
    let direction;
    switch (preset) {
      case "front":
        direction = metrics.axes.depth.clone().multiplyScalar(1.2).add(metrics.axes.height.clone().multiplyScalar(0.18));
        break;
      case "left":
        direction = metrics.axes.width.clone().multiplyScalar(-1.2).add(metrics.axes.height.clone().multiplyScalar(0.18));
        break;
      case "top":
        direction = metrics.axes.height.clone().multiplyScalar(1.35).add(metrics.axes.depth.clone().multiplyScalar(0.18));
        break;
      default:
        direction = metrics.axes.width
          .clone()
          .multiplyScalar(-1.05)
          .add(metrics.axes.depth.clone().multiplyScalar(1.15))
          .add(metrics.axes.height.clone().multiplyScalar(0.72));
    }

    direction.normalize();
    camera.up.copy(metrics.axes.height);
    camera.position.copy(metrics.center.clone().add(direction.multiplyScalar(distance)));
    camera.near = 0.05;
    camera.far = Math.max(distance * 12, 50);
    camera.lookAt(metrics.center);
    camera.updateProjectionMatrix();
    if (controlsRef.current) {
      controlsRef.current.target.copy(metrics.center);
      controlsRef.current.update();
    }
  }, [camera, controlsRef, metrics, preset]);

  return null;
}

function ClippingController({ enabled }) {
  const { gl } = useThree();

  useEffect(() => {
    gl.localClippingEnabled = enabled;
    return () => {
      gl.localClippingEnabled = false;
    };
  }, [enabled, gl]);

  return null;
}

function DenseMesh({ url }) {
  const geometry = useLoader(PLYLoader, url);
  const meshRef = useRef(null);

  useEffect(() => {
    if (!geometry) return;
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }, [geometry]);

  if (!geometry) return null;
  const hasColors = geometry.hasAttribute("color");
  return (
    <mesh ref={meshRef} geometry={geometry} receiveShadow>
      <meshStandardMaterial
        vertexColors={hasColors}
        color={hasColors ? undefined : "#b0b8c4"}
        transparent
        opacity={0.16}
        roughness={0.85}
        metalness={0.05}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function SceneContent({
  rack,
  shelves,
  allProducts,
  layoutConfig,
  visibleProducts,
  selectedProduct,
  selectedShelfId,
  viewPreset,
  viewerOptions,
  viewportEditingDisabled,
  onSelectProduct,
  onSelectShelf,
  onCommitReferenceWidth,
  onCommitShelfOffset,
  denseMeshUrl
}) {
  const { camera, gl } = useThree();
  const controlsRef = useRef(null);
  const draftWidthRef = useRef(null);
  const draftShelfOffsetsRef = useRef({});
  const [draftWidth, setDraftWidth] = useState(null);
  const [draftShelfOffsets, setDraftShelfOffsets] = useState({});
  const [dragState, setDragState] = useState(null);
  // Use allProducts (not visibleProducts) so the bounding box — and therefore the
  // camera preset position — doesn't change when the user filters by shelf.
  const metrics = useMemo(() => buildSceneMetrics(rack, shelves, allProducts), [rack, shelves, allProducts]);
  const selectedShelf = useMemo(
    () => shelves.find((shelf) => shelf.id === selectedShelfId) ?? null,
    [selectedShelfId, shelves]
  );
  const appliedRackWidth = useMemo(
    () => (rack?.bounds ? rack.bounds.width[1] - rack.bounds.width[0] : null),
    [rack]
  );
  const appliedShelfOffsets = layoutConfig?.shelf_height_offsets_m ?? {};
  const previewRackDimensions = useMemo(
    () => buildRackDimensionSet(rack, draftWidth ?? appliedRackWidth ?? undefined),
    [appliedRackWidth, draftWidth, rack]
  );
  const selectedShelfAppliedOffset = selectedShelf
    ? Number(appliedShelfOffsets[selectedShelf.id] ?? 0)
    : 0;
  const selectedShelfPreviewOffset = selectedShelf
    ? draftShelfOffsets[selectedShelf.id] ?? selectedShelfAppliedOffset
    : 0;
  const selectedShelfDimensions = useMemo(
    () =>
      selectedShelf
        ? buildShelfDimensionSet(selectedShelf, selectedShelfPreviewOffset - selectedShelfAppliedOffset)
        : null,
    [selectedShelf, selectedShelfAppliedOffset, selectedShelfPreviewOffset]
  );
  const highlightedShelfId = selectedProduct?.shelf_id ?? null;
  const shelfById = useMemo(
    () => new Map(shelves.map((shelf) => [shelf.id, shelf])),
    [shelves]
  );
  const floorSize = Math.max(metrics.size.x, metrics.size.z, metrics.radius * 2.8);
  const debugRayOrigin = useMemo(() => buildDebugRayOrigin(rack), [rack]);
  const instructionPosition = useMemo(
    () =>
      metrics.center
        .clone()
        .add(metrics.axes.height.clone().multiplyScalar(metrics.size.y * 0.46))
        .add(metrics.axes.depth.clone().multiplyScalar(metrics.size.z * 0.02)),
    [metrics]
  );
  const clippingPlanes = useMemo(() => {
    if (!viewerOptions.section) {
      return [];
    }

    const cutCenter = metrics.center
      .clone()
      .add(metrics.axes.depth.clone().multiplyScalar(metrics.size.z * 0.08));
    const normal = metrics.axes.depth.clone().normalize().multiplyScalar(-1);
    return [new THREE.Plane().setFromNormalAndCoplanarPoint(normal, cutCenter)];
  }, [metrics, viewerOptions.section]);
  const beginWidthDrag = useCallback(
    (control) => {
      if (viewportEditingDisabled || !control || !Number.isFinite(appliedRackWidth)) {
        return;
      }

      setDragState({
        kind: "width",
        axis: control.axis,
        linePoint: control.linePoint,
        initialValue: appliedRackWidth
      });
      setDraftWidth(appliedRackWidth);
    },
    [appliedRackWidth, viewportEditingDisabled]
  );
  const beginShelfOffsetDrag = useCallback(
    (shelfId, handle) => {
      if (viewportEditingDisabled || !handle) {
        return;
      }

      const currentOffset = Number(appliedShelfOffsets[shelfId] ?? 0);
      setDragState({
        kind: "shelf-offset",
        shelfId,
        axis: handle.axis,
        linePoint: handle.linePoint,
        initialValue: currentOffset
      });
      setDraftShelfOffsets((current) => ({
        ...current,
        [shelfId]: currentOffset
      }));
    },
    [appliedShelfOffsets, viewportEditingDisabled]
  );

  useEffect(() => {
    draftWidthRef.current = draftWidth;
  }, [draftWidth]);

  useEffect(() => {
    draftShelfOffsetsRef.current = draftShelfOffsets;
  }, [draftShelfOffsets]);

  useEffect(() => {
    setDragState(null);
    setDraftWidth(null);
    setDraftShelfOffsets({});
    draftWidthRef.current = null;
    draftShelfOffsetsRef.current = {};
    document.body.style.cursor = "";
  }, [layoutConfig, rack, shelves]);

  useEffect(() => {
    if (!dragState) {
      return undefined;
    }

    function handlePointerMove(event) {
      const projectedPoint = projectPointerToAxis(
        event.clientX,
        event.clientY,
        dragState.linePoint,
        dragState.axis,
        camera,
        gl.domElement
      );

      if (!projectedPoint) {
        return;
      }

      const axisDirection = dragState.axis.clone().normalize();
      const scalar = projectedPoint.clone().sub(dragState.linePoint).dot(axisDirection);

      if (dragState.kind === "width") {
        const nextWidth = Math.max(0.5, snapValue(Math.abs(scalar) * 2, 0.05));
        setDraftWidth(nextWidth);
        return;
      }

      const nextOffset = snapValue(dragState.initialValue + scalar, 0.01);
      setDraftShelfOffsets((current) => ({
        ...current,
        [dragState.shelfId]: nextOffset
      }));
    }

    function handlePointerUp() {
      document.body.style.cursor = "";

      if (dragState.kind === "width") {
        const nextWidth = draftWidthRef.current;
        if (Number.isFinite(nextWidth) && Math.abs(nextWidth - dragState.initialValue) >= 0.01) {
          onCommitReferenceWidth?.(nextWidth);
        }
      } else {
        const nextOffset = draftShelfOffsetsRef.current[dragState.shelfId];
        if (Number.isFinite(nextOffset) && Math.abs(nextOffset - dragState.initialValue) >= 0.005) {
          onCommitShelfOffset?.(dragState.shelfId, nextOffset);
        }
      }

      setDragState(null);
      setDraftWidth(null);
      setDraftShelfOffsets({});
      draftWidthRef.current = null;
      draftShelfOffsetsRef.current = {};
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp, { once: true });
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [camera, dragState, gl.domElement, onCommitReferenceWidth, onCommitShelfOffset]);

  const { invalidate } = useThree();

  return (
    <>
      <InvalidateOnChange values={[
        viewPreset, viewerOptions, selectedProduct, selectedShelfId, visibleProducts,
        draftWidth, draftShelfOffsets, rack, shelves,
      ]} />
      <ClippingController enabled={viewerOptions.section} />
      <ViewPresetController metrics={metrics} preset={viewPreset} controlsRef={controlsRef} />
      <ambientLight intensity={0.96} />
      <hemisphereLight skyColor="#c7cbd1" groundColor="#4b5058" intensity={0.62} />
      <directionalLight
        position={metrics.center.clone().add(new THREE.Vector3(6, 9, 4))}
        intensity={1.1}
        color="#ffffff"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={metrics.center.clone().add(new THREE.Vector3(-5, 5, -4))} intensity={0.28} color="#d1d5db" />
      <directionalLight position={metrics.center.clone().add(new THREE.Vector3(2, 3, 7))} intensity={0.24} color="#e5e7eb" />

      <gridHelper
        args={[floorSize * 0.96, 30, "#6b7078", "#5b6068"]}
        position={metrics.center.clone().add(metrics.axes.height.clone().multiplyScalar(-0.014))}
      />
      <SceneInstruction position={instructionPosition} productCount={allProducts.length} />

      {viewerOptions.points && denseMeshUrl ? (
        <Suspense fallback={null}>
          <DenseMesh url={denseMeshUrl} />
        </Suspense>
      ) : null}
      {shelves.length === 0 && <EmptyScene />}
      {viewerOptions.points &&
        shelves.map((shelf, shelfIndex) => {
          const emphasized = selectedShelfId === shelf.id || highlightedShelfId === shelf.id;
          const dimmed = Boolean(selectedShelfId || highlightedShelfId) && !emphasized;
          return (
            <ShelfClusterPoints
              key={`${shelf.id}-points`}
              shelf={shelf}
              shelfIndex={shelfIndex}
              highlighted={emphasized}
              dimmed={dimmed}
              clippingPlanes={clippingPlanes}
            />
          );
        })}
      <RackGeometry rack={rack} clippingPlanes={clippingPlanes} xray={viewerOptions.xray} />
      {shelves.map((shelf) => (
        <group key={shelf.id}>
          <ShelfMesh
            shelf={shelf}
            highlighted={selectedShelfId === shelf.id || highlightedShelfId === shelf.id}
            wireframe={viewerOptions.wireframe}
            xray={viewerOptions.xray}
            clippingPlanes={clippingPlanes}
            onSelect={onSelectShelf}
          />
          <ShelfFrontLip
            shelf={shelf}
            highlighted={selectedShelfId === shelf.id || highlightedShelfId === shelf.id}
            clippingPlanes={clippingPlanes}
            xray={viewerOptions.xray}
          />
        </group>
      ))}
      {viewerOptions.dimensions &&
        (selectedShelf ? (
          <ShelfDimensions dimensionSet={selectedShelfDimensions} />
        ) : (
          <RackDimensions dimensionSet={previewRackDimensions} />
        ))}
      {viewerOptions.dimensions && previewRackDimensions?.handles?.width ? (
        <group>
          <ViewportHandle
            position={previewRackDimensions.handles.width.minPosition}
            active={dragState?.kind === "width"}
            disabled={viewportEditingDisabled}
            color="#7dd3fc"
            label={dragState?.kind === "width" ? formatDimensionLength(draftWidth ?? appliedRackWidth ?? 0) : "Width"}
            onPointerDown={() => beginWidthDrag(previewRackDimensions.handles.width)}
          />
          <ViewportHandle
            position={previewRackDimensions.handles.width.maxPosition}
            active={dragState?.kind === "width"}
            disabled={viewportEditingDisabled}
            color="#7dd3fc"
            label={dragState?.kind === "width" ? formatDimensionLength(draftWidth ?? appliedRackWidth ?? 0) : "Width"}
            onPointerDown={() => beginWidthDrag(previewRackDimensions.handles.width)}
          />
        </group>
      ) : null}
      {viewerOptions.dimensions && selectedShelf && selectedShelfDimensions?.handle ? (
        <ViewportHandle
          position={selectedShelfDimensions.handle.position}
          active={dragState?.kind === "shelf-offset" && dragState.shelfId === selectedShelf.id}
          disabled={viewportEditingDisabled}
          color="#f8c34f"
          label={
            dragState?.kind === "shelf-offset" && dragState.shelfId === selectedShelf.id
              ? `Offset ${selectedShelfPreviewOffset >= 0 ? "+" : "-"}${formatDimensionLength(
                  Math.abs(selectedShelfPreviewOffset)
                )}`
              : "Height"
          }
          onPointerDown={() => beginShelfOffsetDrag(selectedShelf.id, selectedShelfDimensions.handle)}
        />
      ) : null}
      {viewerOptions.anchors && (
        <>
          <MarkersLayer
            products={visibleProducts}
            shelves={shelves}
            selectedId={selectedProduct?.id ?? null}
            onSelect={onSelectProduct}
            clippingPlanes={clippingPlanes}
          />
          {selectedProduct && (
            <ProductMarker
              key={selectedProduct.id}
              product={selectedProduct}
              shelf={shelfById.get(selectedProduct.shelf_id)}
              active
              rayOrigin={debugRayOrigin}
              showRay={viewerOptions.rays}
              onSelect={onSelectProduct}
              clippingPlanes={clippingPlanes}
            />
          )}
        </>
      )}

      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        maxPolarAngle={Math.PI * 0.495}
        minDistance={0.6}
        maxDistance={Math.max(metrics.radius * 6, 8)}
        onChange={invalidate}
      />
      <GizmoHelper alignment="bottom-right" margin={[96, 96]}>
        <GizmoViewport axisColors={["#f87171", "#86efac", "#7dd3fc"]} labelColor="#dbeafe" />
      </GizmoHelper>
    </>
  );
}

export default function SceneViewer({
  results,
  layoutConfig,
  selectedProduct,
  selectedShelfId,
  viewPreset,
  viewerOptions,
  viewportEditingDisabled,
  onSelectProduct,
  onSelectShelf,
  onCommitReferenceWidth,
  onCommitShelfOffset
}) {
  const shelves = results?.shelves ?? [];
  const products = results?.products ?? [];
  const rack = results?.rack ?? null;
  const denseMeshUrl = results?.dense_mesh_url ?? null;
  const visibleProducts = selectedShelfId
    ? products.filter((product) => product.shelf_id === selectedShelfId)
    : products;

  return (
    <Canvas
      camera={{ position: [3.2, 2.3, 3.0], fov: 38 }}
      shadows
      dpr={[1, 1.5]}
      style={{
        width: "100%",
        height: "100%",
        display: "block",
        background: "linear-gradient(180deg, rgba(28, 37, 48, 0.98), rgba(10, 16, 24, 1))"
      }}
    >
      <color attach="background" args={["#565b63"]} />
      <fog attach="fog" args={["#565b63", 12, 26]} />
      <SceneContent
        rack={rack}
        shelves={shelves}
        allProducts={products}
        layoutConfig={layoutConfig}
        visibleProducts={visibleProducts}
        selectedProduct={selectedProduct}
        selectedShelfId={selectedShelfId}
        viewPreset={viewPreset}
        viewerOptions={viewerOptions}
        viewportEditingDisabled={viewportEditingDisabled}
        onSelectProduct={onSelectProduct}
        onSelectShelf={onSelectShelf}
        onCommitReferenceWidth={onCommitReferenceWidth}
        onCommitShelfOffset={onCommitShelfOffset}
        denseMeshUrl={denseMeshUrl}
      />
    </Canvas>
  );
}
