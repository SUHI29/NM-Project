const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const mongoose = require('mongoose');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

// Initialize Express app and HTTP server
const app = express();
const server = http.createServer(app);
const allowedOrigins = process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : ['http://localhost:3000', 'http://localhost:8080'];
const io = new Server(server, {
  cors: {
    origin: allowedOrigins,
    methods: ['GET', 'POST']
  }
});

// Middleware
app.use(cors({
  origin: allowedOrigins,
  methods: ['GET', 'POST']
}));
app.use(express.json());

// Log environment variables (redact API keys)
console.log('Environment variables:', {
  PORT: process.env.PORT,
  CORS_ORIGINS: process.env.CORS_ORIGINS,
  AIRNOW_API_KEY: process.env.AIRNOW_API_KEY ? `****${process.env.AIRNOW_API_KEY.slice(-4)}` : 'Not set',
  IQAIR_API_KEY: process.env.IQAIR_API_KEY ? `****${process.env.IQAIR_API_KEY.slice(-4)}` : 'Not set'
});

// MongoDB Connection
mongoose.connect('mongodb://localhost:27017/airQualityDB', {
  useUnifiedTopology: true,
  useNewUrlParser: true
})
  .then(() => {
    console.log('Connected to MongoDB successfully');
  })
  .catch(err => {
    console.error('MongoDB connection error:', err.message, err.stack);
  });
mongoose.set('debug', true); // Enable MongoDB debug logging

// Air Quality Data Schema
const airQualitySchema = new mongoose.Schema({
  location: String,
  aqi: Number,
  pm25: { type: Number, default: 0 },
  pm10: { type: Number, default: 0 },
  o3: { type: Number, default: 0 },
  no2: { type: Number, default: 0 },
  so2: { type: Number, default: 0 },
  wind: Number,
  humidity: Number,
  temperature: Number,
  pressure: Number,
  timestamp: { type: Date, default: Date.now }
});

const AirQuality = mongoose.model('AirQuality', airQualitySchema);

// Function to fetch real-time air quality data from IQAir (primary) or AirNow (fallback)
async function fetchAirQualityData() {
  console.log('Fetching air quality data...');

  // Try IQAir API first
  if (process.env.IQAIR_API_KEY) {
    console.log('Attempting IQAir API...');
    const iqairData = await fetchIQAirData();
    if (iqairData && iqairData.location !== 'New York City (Mock)') {
      return iqairData;
    }
    console.log('IQAir API failed, falling back to AirNow...');
  } else {
    console.warn('IQAIR_API_KEY is not set in .env');
  }

  // Fallback to AirNow API
  if (process.env.AIRNOW_API_KEY) {
    console.log('Attempting AirNow API...');
    const airnowData = await fetchAirNowData();
    return airnowData;
  } else {
    console.error('AIRNOW_API_KEY is not set in .env');
    return mockData();
  }
}

// Fetch data from IQAir API
async function fetchIQAirData() {
  const url = 'http://api.airvisual.com/v2/nearest_city';
  const params = {
    lat: 40.7128, // New York City
    lon: -74.0060,
    key: process.env.IQAIR_API_KEY
  };
  console.log('IQAir API request:', { url, params: { ...params, key: `****${params.key.slice(-4)}` } });

  try {
    const response = await axios.get(url, { params });
    console.log('IQAir API response:', {
      status: response.status,
      data: response.data,
      headers: response.headers
    });

    const data = response.data.data;
    if (!data.current || !data.current.pollution) {
      console.warn('No pollution data returned from IQAir API');
      return mockData();
    }

    const pollution = data.current.pollution;
    const weather = data.current.weather || {};

    const result = {
      location: data.city || 'New York City',
      aqi: pollution.aqius || 0,
      pm25: pollution.p2 && pollution.p2.conc != null ? pollution.p2.conc : 0,
      pm10: pollution.p1 && pollution.p1.conc != null ? pollution.p1.conc : 0,
      o3: pollution.o3 && pollution.o3.conc != null ? pollution.o3.conc : 0,
      no2: pollution.n2 && pollution.n2.conc != null ? pollution.n2.conc : 0,
      so2: pollution.s2 && pollution.s2.conc != null ? pollution.s2.conc : 0,
      wind: weather.ws || Math.random() * 20,
      humidity: weather.hu || Math.random() * 100,
      temperature: weather.tp != null ? weather.tp : 20 + Math.random() * 10,
      pressure: weather.pr || 1000 + Math.random() * 20
    };

    console.log('Processed IQAir data:', result);
    return result;
  } catch (error) {
    console.error('Error fetching air quality data from IQAir:', {
      message: error.message,
      status: error.response?.status,
      headers: error.response?.headers,
      data: error.response?.data,
      stack: error.stack
    });
    return null;
  }
}

// Fetch data from AirNow API
async function fetchAirNowData() {
  if (!process.env.AIRNOW_API_KEY) {
    console.error('AIRNOW_API_KEY is not set in .env');
    return mockData();
  }

  const url = 'https://www.airnowapi.org/aq/data/';
  const maxRetries = 3;

  // Try coordinates first
  const coordParams = {
    parameters: 'PM25,PM10,O3,NO2,SO2',
    lat: 40.7128,
    lon: -74.0060,
    API_KEY: process.env.AIRNOW_API_KEY
  };
  console.log('AirNow API request (coordinates):', { url, params: { ...coordParams, API_KEY: `****${coordParams.API_KEY.slice(-4)}` } });

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await axios.get(url, { params: coordParams });
      console.log('AirNow API response (coordinates, attempt', attempt, '):', {
        status: response.status,
        data: response.data,
        headers: response.headers
      });
      return processAirNowResponse(response.data);
    } catch (error) {
      console.error('Error fetching air quality data from AirNow (coordinates, attempt', attempt, '):', {
        message: error.message,
        status: error.response?.status,
        headers: error.response?.headers,
        data: error.response?.data,
        stack: error.stack
      });
      if (attempt === maxRetries || error.response?.status !== 500) {
        console.log('Exhausted retries or non-500 error for coordinates, trying ZIP code...');
        break;
      }
      console.log('Retrying coordinates request...');
    }
  }

  // Fallback to ZIP code
  const zipParams = {
    parameters: 'PM25,PM10,O3,NO2,SO2',
    zipCode: '10001',
    API_KEY: process.env.AIRNOW_API_KEY
  };
  console.log('AirNow API request (ZIP code):', { url, params: { ...zipParams, API_KEY: `****${zipParams.API_KEY.slice(-4)}` } });

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await axios.get(url, { params: zipParams });
      console.log('AirNow API response (ZIP code, attempt', attempt, '):', {
        status: response.status,
        data: response.data,
        headers: response.headers
      });
      return processAirNowResponse(response.data);
    } catch (error) {
      console.error('Error fetching air quality data from AirNow (ZIP code, attempt', attempt, '):', {
        message: error.message,
        status: error.response?.status,
        headers: error.response?.headers,
        data: error.response?.data,
        stack: error.stack
      });
      if (attempt === maxRetries) {
        console.log('Exhausted retries for ZIP code, falling back to mock data...');
        return mockData();
      }
      console.log('Retrying ZIP code request...');
    }
  }

  return mockData();
}

// Process AirNow API response
function processAirNowResponse(measurements) {
  if (!measurements || measurements.length === 0) {
    console.warn('No data returned from AirNow API');
    return mockData();
  }

  let maxAQI = 0;
  const data = {
    location: measurements[0].ReportingArea || 'New York City',
    pm25: 0,
    pm10: 0,
    o3: 0,
    no2: 0,
    so2: 0
  };

  measurements.forEach(measurement => {
    if (measurement.AQI > maxAQI) maxAQI = measurement.AQI;
    switch (measurement.ParameterName) {
      case 'PM2.5':
        data.pm25 = measurement.Value;
        break;
      case 'PM10':
        data.pm10 = measurement.Value;
        break;
      case 'O3':
        data.o3 = measurement.Value;
        break;
      case 'NO2':
        data.no2 = measurement.Value;
        break;
      case 'SO2':
        data.so2 = measurement.Value;
        break;
    }
  });

  const result = {
    location: data.location,
    aqi: maxAQI,
    pm25: data.pm25,
    pm10: data.pm10,
    o3: data.o3,
    no2: data.no2,
    so2: data.so2,
    wind: Math.random() * 20,
    humidity: Math.random() * 100,
    temperature: 20 + Math.random() * 10,
    pressure: 1000 + Math.random() * 20
  };

  console.log('Processed AirNow data:', result);
  return result;
}

// Mock data fallback
function mockData() {
  const mock = {
    location: 'New York City (Mock)',
    aqi: Math.round(50 + Math.random() * 100),
    pm25: Math.random() * 50,
    pm10: Math.random() * 100,
    o3: Math.random() * 200,
    no2: Math.random() * 50,
    so2: Math.random() * 20,
    wind: Math.random() * 20,
    humidity: Math.random() * 100,
    temperature: 20 + Math.random() * 10,
    pressure: 1000 + Math.random() * 20
  };
  console.log('Using mock data due to API failure:', mock);
  return mock;
}

// Simple AQI calculation (optional, as APIs provide AQI directly)
function calculateAQI(measurements) {
  console.log('Calculating AQI from measurements:', measurements);
  const pm25 = measurements.find(m => m.ParameterName === 'PM2.5')?.Value || 0;
  if (pm25 <= 12) return Math.round(pm25 * 4.17);
  if (pm25 <= 35.4) return Math.round(50 + (pm25 - 12) * 2.17);
  if (pm25 <= 55.4) return Math.round(100 + (pm25 - 35.4) * 2.5);
  return 150; // Add more ranges as needed
}

// WebSocket: Push real-time updates to clients
io.on('connection', async (socket) => {
  console.log('Client connected:', socket.id);

  // Send initial data
  try {
    const data = await AirQuality.findOne().sort({ timestamp: -1 }).exec();
    if (data) {
      console.log('Sending initial air quality data to client:', data);
      socket.emit('airQualityUpdate', data);
    } else {
      console.warn('No initial air quality data found in MongoDB');
    }
  } catch (err) {
    console.error('Error fetching initial air quality data from MongoDB:', err.message, err.stack);
  }

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Periodic data fetching (every 5 minutes to respect API rate limits)
console.log('Starting periodic air quality data fetch interval (every 5 minutes)...');
setInterval(async () => {
  console.log('Periodic air quality data fetch triggered at:', new Date().toISOString());
  const airQualityData = await fetchAirQualityData();
  if (!airQualityData) {
    console.warn('No air quality data returned from fetchAirQualityData');
    return;
  }

  // Save to MongoDB
  try {
    const airQuality = new AirQuality(airQualityData);
    await airQuality.save();
    console.log('Saved air quality data to MongoDB:', airQualityData);
  } catch (err) {
    console.error('Error saving air quality data to MongoDB:', err.message, err.stack);
  }

  // Broadcast to all connected clients
  console.log('Broadcasting air quality data to clients:', airQualityData);
  io.emit('airQualityUpdate', airQualityData);
}, 300000); // Update every 5 minutes

// Initial API call on startup
console.log('Triggering initial air quality API call on startup...');
fetchAirQualityData().then(async (airQualityData) => {
  if (!airQualityData) {
    console.warn('No air quality data returned from initial fetchAirQualityData');
    return;
  }
  try {
    const airQuality = new AirQuality(airQualityData);
    await airQuality.save();
    console.log('Saved initial air quality data to MongoDB:', airQualityData);
    io.emit('airQualityUpdate', airQualityData);
    console.log('Broadcasted initial air quality data to clients:', airQualityData);
  } catch (err) {
    console.error('Error saving initial air quality data to MongoDB:', err.message, err.stack);
  }
}).catch(err => {
  console.error('Error during initial air quality API call:', err.message, err.stack);
});

// API Endpoints
app.get('/api/air-quality', async (req, res) => {
  console.log('Received request for /api/air-quality');
  try {
    const data = await AirQuality.find().sort({ timestamp: -1 }).limit(10);
    console.log('Returning air quality data:', data);
    res.json(data);
  } catch (error) {
    console.error('Error fetching air quality data for /api/air-quality:', error.message, error.stack);
    res.status(500).json({ error: 'Error fetching air quality data' });
  }
});

app.get('/api/air-quality/historical', async (req, res) => {
  console.log('Received request for /api/air-quality/historical');
  try {
    const data = await AirQuality.aggregate([
      {
        $group: {
          _id: {
            $dateToString: { format: '%Y-%m', date: '$timestamp' }
          },
          avgAQI: { $avg: '$aqi' },
          avgPM25: { $avg: '$pm25' }
        }
      },
      { $sort: { '_id': -1 } },
      { $limit: 12 }
    ]);
    console.log('Returning historical air quality data:', data);
    res.json(data);
  } catch (error) {
    console.error('Error fetching historical data for /api/air-quality/historical:', error.message, error.stack);
    res.status(500).json({ error: 'Error fetching historical data' });
  }
});

// Manual test endpoint for air quality API
app.get('/api/test-air-quality', async (req, res) => {
  console.log('Received request for /api/test-air-quality');
  try {
    const data = await fetchAirQualityData();
    console.log('Test air quality API result:', data);
    res.json(data);
  } catch (error) {
    console.error('Error testing air quality API:', error.message, error.stack);
    res.status(500).json({ error: 'Error testing air quality API', details: error.message });
  }
});

// Manual test endpoint for MongoDB
app.get('/api/test-mongodb', async (req, res) => {
  console.log('Received request for /api/test-mongodb');
  try {
    const testData = new AirQuality({
      location: 'Test Location',
      aqi: 50,
      pm25: 10,
      pm10: 20,
      o3: 30,
      no2: 15,
      so2: 5,
      wind: 10,
      humidity: 80,
      temperature: 25,
      pressure: 1015
    });
    await testData.save();
    console.log('Test data saved to MongoDB:', testData);
    res.json({ message: 'Test data saved', data: testData });
  } catch (error) {
    console.error('Error saving test data to MongoDB:', error.message, error.stack);
    res.status(500).json({ error: 'Error saving test data', details: error.message });
  }
});

// Start server
const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});