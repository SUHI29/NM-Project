
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';

const AQITrends = () => {
  // Mock data for AQI trends
  const monthlyData = [
    { month: 'Jan', urban: 85, suburban: 65, rural: 45 },
    { month: 'Feb', urban: 82, suburban: 63, rural: 42 },
    { month: 'Mar', urban: 78, suburban: 58, rural: 40 },
    { month: 'Apr', urban: 72, suburban: 55, rural: 38 },
    { month: 'May', urban: 68, suburban: 52, rural: 36 },
    { month: 'Jun', urban: 74, suburban: 57, rural: 39 },
    { month: 'Jul', urban: 80, suburban: 62, rural: 42 },
    { month: 'Aug', urban: 84, suburban: 64, rural: 44 },
    { month: 'Sep', urban: 78, suburban: 61, rural: 42 },
    { month: 'Oct', urban: 76, suburban: 58, rural: 40 },
    { month: 'Nov', urban: 79, suburban: 60, rural: 41 },
    { month: 'Dec', urban: 84, suburban: 65, rural: 45 },
  ];
  
  const yearlyData = [
    { year: '2018', urban: 88, suburban: 68, rural: 48 },
    { year: '2019', urban: 85, suburban: 65, rural: 46 },
    { year: '2020', urban: 75, suburban: 58, rural: 40 },
    { year: '2021', urban: 79, suburban: 61, rural: 42 },
    { year: '2022', urban: 82, suburban: 63, rural: 44 },
    { year: '2023', urban: 80, suburban: 62, rural: 43 },
    { year: '2024', urban: 78, suburban: 60, rural: 42 },
  ];

  return (
    <Card className="glass-panel glow">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xl">Historical AQI Trends</CardTitle>
        <Button variant="outline" size="sm">
          <Download className="h-4 w-4 mr-2" />
          Export Data
        </Button>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="monthly" className="w-full">
          <TabsList className="grid grid-cols-2 mb-6 w-[200px]">
            <TabsTrigger value="monthly">Monthly</TabsTrigger>
            <TabsTrigger value="yearly">Yearly</TabsTrigger>
          </TabsList>
          
          <TabsContent value="monthly" className="mt-0">
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={monthlyData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <XAxis dataKey="month" stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '8px', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <Legend />
                  <Bar dataKey="urban" name="Urban" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="suburban" name="Suburban" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="rural" name="Rural" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="yearly" className="mt-0">
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={yearlyData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <XAxis dataKey="year" stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#8E9196" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderRadius: '8px', border: '1px solid #334155' }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <Legend />
                  <Bar dataKey="urban" name="Urban" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="suburban" name="Suburban" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="rural" name="Rural" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default AQITrends;
