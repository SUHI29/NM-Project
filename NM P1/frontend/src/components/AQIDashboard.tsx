import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import AirQualityMap from './AirQualityMap';
import AirQualityMetrics from './AirQualityMetrics';
import PredictionSection from './PredictionSection';
import PollutantBreakdown from './PollutantBreakdown';
import AQITrends from './AQITrends';
import DashboardHeader from './DashboardHeader';
import ParticleBackground from './ParticleBackground';

const AQIDashboard = () => {
  const [airQualityData, setAirQualityData] = useState(null);

  useEffect(() => {
    // Connect to WebSocket server
    const socket = io('http://localhost:5000');

    socket.on('airQualityUpdate', (data) => {
      setAirQualityData(data);
    });

    // Fetch initial data
    fetch('http://localhost:5000/api/air-quality')
      .then(res => res.json())
      .then(data => setAirQualityData(data[0]))
      .catch(err => console.error('Error fetching initial data:', err));

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <ParticleBackground />
      <div className="container relative z-10 mx-auto px-4 py-8">
        <DashboardHeader />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
          <div className="lg:col-span-2">
            <AirQualityMap airQualityData={airQualityData} />
          </div>
          <div>
            <AirQualityMetrics airQualityData={airQualityData} />
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <PredictionSection />
          <PollutantBreakdown airQualityData={airQualityData} />
        </div>
        <div className="mt-6">
          <AQITrends />
        </div>
      </div>
    </div>
  );
};

export default AQIDashboard;