import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const PollutantBreakdown = ({ airQualityData }) => {
  if (!airQualityData) {
    return <div>Loading pollutant breakdown...</div>;
  }

  // Dynamic data based on real-time airQualityData
  const data = [
    { name: 'PM2.5', value: airQualityData.pm25, color: '#0ea5e9' },
    { name: 'PM10', value: airQualityData.pm10, color: '#8b5cf6' },
    { name: 'O₃', value: airQualityData.o3, color: '#10b981' },
    { name: 'NO₂', value: airQualityData.no2, color: '#f97316' },
    { name: 'SO₂', value: airQualityData.so2, color: '#f43f5e' },
  ];

  // Calculate total to display percentage
  const total = data.reduce((acc, curr) => acc + curr.value, 0) || 1; // Avoid division by zero
  const percentData = data.map(item => ({
    ...item,
    percent: Math.round((item.value / total) * 100)
  }));

  return (
    <Card className="glass-panel glow">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Pollutant Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col md:flex-row items-center">
          <div className="w-full md:w-1/2 h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={percentData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={70}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {percentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value, name) => [`${value}%`, name]}
                  contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '8px', border: '1px solid #334155' }}
                  labelStyle={{ color: '#f8fafc' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="w-full md:w-1/2 space-y-2 mt-4 md:mt-0">
            {percentData.map((item, index) => (
              <div key={index} className="flex items-center">
                <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: item.color }}></div>
                <div className="text-sm">{item.name}</div>
                <div className="ml-auto font-semibold text-sm">{item.percent}%</div>
              </div>
            ))}
          </div>
        </div>
        <div className="data-line my-6"></div>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-muted/20 p-3 rounded-lg">
            <div className="flex justify-between items-center">
              <div className="flex items-center">
                <div className="w-2 h-8 bg-primary rounded-full mr-2"></div>
                <div>
                  <div className="text-sm font-medium">PM2.5</div>
                  <div className="text-xs text-muted-foreground">Fine Particulate Matter</div>
                </div>
              </div>
              <div className="text-lg font-semibold">{airQualityData.pm25} μg/m³</div>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              <span className={airQualityData.pm25 <= 35 ? 'text-aqi-moderate' : 'text-aqi-unhealthy'}>
                {airQualityData.pm25 <= 35 ? 'Moderate' : 'Unhealthy'}
              </span> - May cause breathing discomfort for sensitive individuals
            </div>
          </div>
          <div className="bg-muted/20 p-3 rounded-lg">
            <div className="flex justify-between items-center">
              <div className="flex items-center">
                <div className="w-2 h-8 bg-secondary rounded-full mr-2"></div>
                <div>
                  <div className="text-sm font-medium">O₃</div>
                  <div className="text-xs text-muted-foreground">Ozone</div>
                </div>
              </div>
              <div className="text-lg font-semibold">{airQualityData.o3} ppb</div>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              <span className={airQualityData.o3 <= 70 ? 'text-aqi-moderate' : 'text-aqi-sensitive'}>
                {airQualityData.o3 <= 70 ? 'Moderate' : 'Unhealthy for Sensitive Groups'}
              </span> - Children and people with respiratory conditions should limit outdoor activity
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PollutantBreakdown;