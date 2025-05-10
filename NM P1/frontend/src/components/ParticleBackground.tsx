
import React, { useEffect, useRef, useState } from 'react';

const ParticleBackground = () => {
  const [particles, setParticles] = useState<{ id: number, size: number, x: string, y: string, delay: number, duration: number }[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Generate random particles
    const createParticles = () => {
      const newParticles = [];
      const count = window.innerWidth < 768 ? 15 : 25; // Fewer particles on mobile
      
      for (let i = 0; i < count; i++) {
        newParticles.push({
          id: i,
          size: Math.random() * 8 + 2, // 2-10px
          x: `${Math.random() * 100}%`,
          y: `${Math.random() * 100}%`,
          delay: Math.random() * 5,
          duration: Math.random() * 15 + 15, // 15-30s
        });
      }
      
      setParticles(newParticles);
    };
    
    createParticles();
    
    // Re-create particles on window resize
    const handleResize = () => {
      if (containerRef.current) {
        createParticles();
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div ref={containerRef} className="fixed inset-0 overflow-hidden pointer-events-none">
      {particles.map(particle => (
        <div
          key={particle.id}
          className="absolute rounded-full opacity-30 animate-particle-float"
          style={{
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            left: particle.x,
            top: particle.y,
            backgroundImage: 'linear-gradient(to right, rgba(14, 165, 233, 0.7), rgba(139, 92, 246, 0.7))',
            boxShadow: '0 0 10px rgba(14, 165, 233, 0.5)',
            animationDelay: `${particle.delay}s`,
            animationDuration: `${particle.duration}s`,
          }}
        />
      ))}
      
      {/* Radial gradient background overlay */}
      <div className="absolute inset-0 bg-gradient-radial from-transparent to-background opacity-80"></div>
      
      {/* Light beam effect */}
      <div className="absolute top-0 left-1/4 w-1/2 h-screen bg-gradient-to-b from-primary/5 via-transparent to-transparent transform -skew-x-12 opacity-20"></div>
    </div>
  );
};

export default ParticleBackground;
