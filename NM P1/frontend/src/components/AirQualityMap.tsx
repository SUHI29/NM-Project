import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

const AirQualityMap = ({ airQualityData }) => {
  // Helper to determine AQI color
  const getAQIColor = (aqi) => {
    if (aqi <= 50) return 'bg-aqi-good';
    if (aqi <= 100) return 'bg-aqi-moderate';
    if (aqi <= 150) return 'bg-aqi-sensitive';
    if (aqi <= 200) return 'bg-aqi-unhealthy';
    if (aqi <= 300) return 'bg-aqi-veryunhealthy';
    return 'bg-aqi-hazardous';
  };

  if (!airQualityData) {
    return <div>Loading map...</div>;
  }

  return (
    <Card className="glass-panel overflow-hidden glow">
      <CardHeader className="pb-2">
        <CardTitle className="flex justify-between items-center">
          <span>Air Quality Map</span>
          <span className="text-sm font-normal text-muted-foreground">Live Data</span>
        </CardTitle>
        <CardDescription>Interactive global air quality visualization</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <div className="relative h-[400px] bg-muted/30 rounded-b-lg overflow-hidden">
          <div className="absolute inset-0 bg-gradient-radial from-transparent to-background/50"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-[80%] h-[80%] opacity-30 rounded-full bg-primary/10 animate-pulse-slow"></div>
            <div className="absolute w-[60%] h-[60%] rounded-full border border-accent/30 animate-breathe"></div>
            <div className="absolute w-[40%] h-[40%] rounded-full border border-secondary/30 animate-breathe animation-delay-1000"></div>
          </div>
          {/* Dynamic city marker based on real-time data */}
          <div
            className={`absolute top-[30%] left-[25%] w-3 h-3 ${getAQIColor(airQualityData.aqi)} rounded-full animate-pulse-slow`}
            title={`${airQualityData.location}: AQI ${airQualityData.aqi}`}
          ></div>
          {/* Legend */}
          <div className="absolute bottom-4 right-4 bg-card/80 backdrop-blur-sm rounded-lg p-2">
            <div className="text-xs font-semibold mb-1">Air Quality Index</div>
            <div className="flex items-center gap-2 text-xs">
              <span className="h-2 w-2 bg-aqi-good rounded-full"></span>
              <span>Good</span>
              <span className="h-2 w-2 bg-aqi-moderate rounded-full ml-2"></span>
              <span>Moderate</span>
              <span className="h-2 w-2 bg-aqi-unhealthy rounded-full ml-2"></span>
              <span>Unhealthy</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AirQualityMap;