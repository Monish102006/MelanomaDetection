import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Icosahedron, Sphere } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function LesionCore() {
  const mesh = useRef<THREE.Mesh>(null);
  const inner = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (mesh.current) {
      mesh.current.rotation.y += delta * 0.14;
      mesh.current.rotation.x += delta * 0.05;
    }
    if (inner.current) inner.current.rotation.y -= delta * 0.22;
  });

  return (
    <group>
      <Icosahedron ref={mesh} args={[1.5, 4]}>
        <meshStandardMaterial
          color="#0d3b4a"
          emissive="#0e7f97"
          emissiveIntensity={0.35}
          roughness={0.25}
          metalness={0.6}
          wireframe
        />
      </Icosahedron>
      <Sphere ref={inner} args={[1.05, 64, 64]}>
        <meshStandardMaterial
          color="#08202e"
          emissive="#1c6fd6"
          emissiveIntensity={0.5}
          roughness={0.15}
          metalness={0.85}
        />
      </Sphere>
      <Sphere args={[1.62, 48, 48]}>
        <meshBasicMaterial color="#3ddcf0" transparent opacity={0.05} side={THREE.BackSide} />
      </Sphere>
    </group>
  );
}

function ParticleField() {
  const points = useRef<THREE.Points>(null);
  const geometry = useMemo(() => {
    const count = 900;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const r = 2.2 + Math.random() * 2.6;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.75;
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, []);

  useFrame((_, delta) => {
    if (points.current) points.current.rotation.y += delta * 0.045;
  });

  return (
    <points ref={points} geometry={geometry}>
      <pointsMaterial size={0.022} color="#7ee9ff" transparent opacity={0.75} sizeAttenuation />
    </points>
  );
}

export default function LesionScene() {
  return (
    <Canvas
      dpr={[1, 1.8]}
      camera={{ position: [0, 0, 6], fov: 45 }}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.35} />
      <pointLight position={[4, 3, 5]} intensity={45} color="#4bd8ef" distance={20} />
      <pointLight position={[-5, -2, -3]} intensity={30} color="#2f6fe0" distance={20} />
      <Float speed={1.1} rotationIntensity={0.35} floatIntensity={0.7}>
        <LesionCore />
      </Float>
      <ParticleField />
    </Canvas>
  );
}
