import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Wind, Droplet, ThermometerSun, AlertCircle } from 'lucide-react';

const AirQualityMetrics = ({ airQualityData }) => {
  // Helper function to determine AQI category and color
  function getAQICategory(aqi) {
    if (aqi <= 50) return { label: 'Good', color: 'bg-aqi-good', textColor: 'text-aqi-good' };
    if (aqi <= 100) return { label: 'Moderate', color: 'bg-aqi-moderate', textColor: 'text-aqi-moderate' };
    if (aqi <= 150) return { label: 'Unhealthy for Sensitive Groups', color: 'bg-aqi-sensitive', textColor: 'text-aqi-sensitive' };
    if (aqi <= 200) return { label: 'Unhealthy', color: 'bg-aqi-unhealthy', textColor: 'text-aqi-unhealthy' };
    if (aqi <= 300) return { label: 'Very Unhealthy', color: 'bg-aqi-veryunhealthy', textColor: 'text-aqi-veryunhealthy' };
    return { label: 'Hazardous', color: 'bg-aqi-hazardous', textColor: 'text-aqi-hazardous' };
  }

  if (!airQualityData) {
    return <div>Loading...</div>;
  }

  const aqiCategory = getAQICategory(airQualityData.aqi);

  return (
    <Card className="glass-panel glow">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Current Air Quality</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center mb-6">
          <div className="relative w-36 h-36 flex items-center justify-center">
            <div className={`absolute inset-0 rounded-full ${aqiCategory.color} opacity-10`}></div>
            <div className="relative text-center">
              <div className="text-5xl font-bold">{airQualityData.aqi}</div>
              <div className={`text-sm font-medium ${aqiCategory.textColor}`}>
                {aqiCategory.label}
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="data-line mb-6"></div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col">
              <div className="flex items-center mb-1">
                <Wind className="h-4 w-4 mr-2 text-primary" />
                <span className="text-xs">Wind</span>
              </div>
              <span className="text-sm font-semibold">{airQualityData.wind} km/h</span>
            </div>

            <div className="flex flex-col">
              <div className="flex items-center mb-1">
                <Droplet className="h-4 w-4 mr-2 text-primary" />
                <span className="text-xs">Humidity</span>
              </div>
              <span className="text-sm font-semibold">{airQualityData.humidity}%</span>
            </div>

            <div className="flex flex-col">
              <div className="flex items-center mb-1">
                <ThermometerSun className="h-4 w-4 mr-2 text-primary" />
                <span className="text-xs">Temperature</span>
              </div>
              <span className="text-sm font-semibold">{airQualityData.temperature}°C</span>
            </div>

            <div className="flex flex-col">
              <div className="flex items-center mb-1">
                <AlertCircle className="h-4 w-4 mr-2 text-primary" />
                <span className="text-xs">Pressure</span>
              </div>
              <span className="text-sm font-semibold">{airQualityData.pressure} hPa</span>
            </div>
          </div>

          <div className="data-line my-6"></div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-xs font-medium">PM2.5</span>
                <span className="text-xs text-muted-foreground">{airQualityData.pm25} μg/m³</span>
              </div>
              <Progress value={airQualityData.pm25 / 2} className="h-2" />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-xs font-medium">PM10</span>
                <span className="text-xs text-muted-foreground">{airQualityData.pm10} μg/m³</span>
              </div>
              <Progress value={airQualityData.pm10 / 3} className="h-2" />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-xs font-medium">O₃</span>
                <span className="text-xs text-muted-foreground">{airQualityData.o3} ppb</span>
              </div>
              <Progress value={airQualityData.o3 / 2} className="h-2" />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AirQualityMetrics;