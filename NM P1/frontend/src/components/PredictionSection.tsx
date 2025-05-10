
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const PredictionSection = () => {
  // Mock data for predictions
  const hourlyData = [
    { time: '06:00', aqi: 45, prediction: 52 },
    { time: '09:00', aqi: 54, prediction: 67 },
    { time: '12:00', aqi: 68, prediction: 75 },
    { time: '15:00', aqi: 62, prediction: 58 },
    { time: '18:00', aqi: 55, prediction: 49 },
    { time: '21:00', aqi: 46, prediction: 43 },
    { time: '00:00', aqi: 40, prediction: 38 },
  ];
  
  const dailyData = [
    { day: 'Mon', aqi: 58, prediction: 63 },
    { day: 'Tue', aqi: 63, prediction: 68 },
    { day: 'Wed', aqi: 72, prediction: 75 },
    { day: 'Thu', aqi: 65, prediction: 60 },
    { day: 'Fri', aqi: 58, prediction: 55 },
    { day: 'Sat', aqi: 55, prediction: 62 },
    { day: 'Sun', aqi: 62, prediction: 68 },
  ];
  
  const weeklyData = [
    { week: 'Week 1', aqi: 55, prediction: 60 },
    { week: 'Week 2', aqi: 62, prediction: 67 },
    { week: 'Week 3', aqi: 68, prediction: 72 },
    { week: 'Week 4', aqi: 62, prediction: 58 },
  ];

  return (
    <Card className="glass-panel glow">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span>Prediction Analysis</span>
          <div className="bg-muted/30 text-xs py-1 px-2 rounded-full">
            ML Model: Random Forest
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="hourly" className="w-full">
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="hourly">Hourly</TabsTrigger>
            <TabsTrigger value="daily">Daily</TabsTrigger>
            <TabsTrigger value="weekly">Weekly</TabsTrigger>
          </TabsList>
          
          <TabsContent value="hourly" className="mt-0">
            <div className="h-[200px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hourlyData} margin={{ top: 5, right: 5, left: -15, bottom: 5 }}>
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorPrediction" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '8px', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <Area type="monotone" dataKey="aqi" stroke="#0ea5e9" fillOpacity={0.3} fill="url(#colorActual)" />
                  <Area type="monotone" dataKey="prediction" stroke="#8b5cf6" fillOpacity={0.3} fill="url(#colorPrediction)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            <div className="flex justify-center mt-4 text-xs">
              <div className="flex items-center mr-4">
                <div className="w-3 h-3 bg-primary rounded-full mr-1"></div>
                <span>Actual</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-secondary rounded-full mr-1"></div>
                <span>Predicted</span>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="daily" className="mt-0">
            <div className="h-[200px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={dailyData} margin={{ top: 5, right: 5, left: -15, bottom: 5 }}>
                  <defs>
                    <linearGradient id="colorActualDaily" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorPredictionDaily" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="day" stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '8px', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <Area type="monotone" dataKey="aqi" stroke="#0ea5e9" fillOpacity={0.3} fill="url(#colorActualDaily)" />
                  <Area type="monotone" dataKey="prediction" stroke="#8b5cf6" fillOpacity={0.3} fill="url(#colorPredictionDaily)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            <div className="flex justify-center mt-4 text-xs">
              <div className="flex items-center mr-4">
                <div className="w-3 h-3 bg-primary rounded-full mr-1"></div>
                <span>Actual</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-secondary rounded-full mr-1"></div>
                <span>Predicted</span>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="weekly" className="mt-0">
            <div className="h-[200px] mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={weeklyData} margin={{ top: 5, right: 5, left: -15, bottom: 5 }}>
                  <defs>
                    <linearGradient id="colorActualWeekly" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorPredictionWeekly" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="week" stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '8px', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <Area type="monotone" dataKey="aqi" stroke="#0ea5e9" fillOpacity={0.3} fill="url(#colorActualWeekly)" />
                  <Area type="monotone" dataKey="prediction" stroke="#8b5cf6" fillOpacity={0.3} fill="url(#colorPredictionWeekly)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            <div className="flex justify-center mt-4 text-xs">
              <div className="flex items-center mr-4">
                <div className="w-3 h-3 bg-primary rounded-full mr-1"></div>
                <span>Actual</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-secondary rounded-full mr-1"></div>
                <span>Predicted</span>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default PredictionSection;
