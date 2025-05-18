import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
    TimeScale,
    TimeSeriesScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
    TimeScale,
    TimeSeriesScale
);

// --- Configuration ---
const API_BASE_URL = 'http://127.0.0.1:8000/api';
const POLLING_INTERVAL = 10 * 60 * 1000; // 10 minutes in milliseconds

const MAJOR_CITIES = [
    { name: "Delhi", city: "Delhi", state: "Delhi", country: "India", lat: 28.6139, lon: 77.2090 },
    { name: "Mumbai", city: "Mumbai", state: "Maharashtra", country: "India", lat: 19.0760, lon: 72.8777 },
    { name: "Bengaluru", city: "Bengaluru", state: "Karnataka", country: "India", lat: 12.9716, lon: 77.5946 },
    { name: "Chennai", city: "Chennai", state: "Tamil Nadu", country: "India", lat: 13.0827, lon: 80.2707 },
    { name: "Kolkata", city: "Kolkata", state: "West Bengal", country: "India", lat: 22.5726, lon: 88.3639 }
];

// --- Helper Functions ---
const getAqiInfo = (aqi) => {
    const recommendationBaseClass = "border-l-4 p-4 rounded-r-md";
    const textBaseClass = "text-gray-200";

    if (aqi === null || aqi === undefined || aqi < 0) return { 
        label: "Unknown", 
        colorClass: "text-gray-400", 
        bgColorClass: "bg-gray-600", 
        borderColorClass: "border-gray-500", 
        glowColor: "gray-500",
        recommendation: "Air quality data is currently unavailable.", 
        recommendationClasses: `${recommendationBaseClass} bg-gray-700 border-gray-500 ${textBaseClass}` 
    };
    if (aqi <= 50) return { 
        label: "Good", 
        colorClass: "text-green-400", 
        bgColorClass: "bg-green-500", 
        borderColorClass: "border-green-400", 
        glowColor: "green-500",
        recommendation: "Air quality is good. It's a great day to be active outside!", 
        recommendationClasses: `${recommendationBaseClass} bg-green-700/30 border-green-500 ${textBaseClass}` 
    };
    if (aqi <= 100) return { 
        label: "Moderate", 
        colorClass: "text-yellow-400", 
        bgColorClass: "bg-yellow-500", 
        borderColorClass: "border-yellow-400", 
        glowColor: "yellow-500",
        recommendation: "Unusually sensitive individuals may experience respiratory symptoms.", 
        recommendationClasses: `${recommendationBaseClass} bg-yellow-700/30 border-yellow-500 ${textBaseClass}` 
    };
    if (aqi <= 150) return { 
        label: "Unhealthy for Sensitive", 
        colorClass: "text-orange-400", 
        bgColorClass: "bg-orange-500", 
        borderColorClass: "border-orange-400", 
        glowColor: "orange-500",
        recommendation: "Members of sensitive groups may experience health effects.", 
        recommendationClasses: `${recommendationBaseClass} bg-orange-700/30 border-orange-500 ${textBaseClass}` 
    };
    if (aqi <= 200) return { 
        label: "Unhealthy", 
        colorClass: "text-red-400", 
        bgColorClass: "bg-red-500", 
        borderColorClass: "border-red-400", 
        glowColor: "red-500",
        recommendation: "Everyone may begin to experience health effects. Reduce prolonged exertion.", 
        recommendationClasses: `${recommendationBaseClass} bg-red-700/30 border-red-500 ${textBaseClass}` 
    };
    if (aqi <= 300) return { 
        label: "Very Unhealthy", 
        colorClass: "text-purple-400", 
        bgColorClass: "bg-purple-500", 
        borderColorClass: "border-purple-400", 
        glowColor: "purple-500",
        recommendation: "Health alert: The risk of health effects is increased for everyone.", 
        recommendationClasses: `${recommendationBaseClass} bg-purple-700/30 border-purple-500 ${textBaseClass}` 
    };
    return { 
        label: "Hazardous", 
        colorClass: "text-fuchsia-400", 
        bgColorClass: "bg-fuchsia-500", 
        borderColorClass: "border-fuchsia-400", 
        glowColor: "fuchsia-500",
        recommendation: "Health warning: Everyone is more likely to be affected. Avoid outdoor activity.", 
        recommendationClasses: `${recommendationBaseClass} bg-fuchsia-700/30 border-fuchsia-500 ${textBaseClass}` 
    };
};

// Convert UTC timestamp to IST and format as DD/MM/YYYY, h:mm:ss a
const formatTimestampToIST = (timestamp) => {
    const date = new Date(timestamp);
    const istOffsetMs = 5.5 * 60 * 60 * 1000;
    const istDate = new Date(date.getTime() + istOffsetMs);

    const day = String(istDate.getUTCDate()).padStart(2, '0');
    const month = String(istDate.getUTCMonth() + 1).padStart(2, '0');
    const year = istDate.getUTCFullYear();

    let hours = istDate.getUTCHours();
    const minutes = String(istDate.getUTCMinutes()).padStart(2, '0');
    const seconds = String(istDate.getUTCSeconds()).padStart(2, '0');
    const ampm = hours >= 12 ? 'pm' : 'am';
    hours = hours % 12 || 12;
    hours = String(hours).padStart(2, '0');

    return `${day}/${month}/${year}, ${hours}:${minutes}:${seconds} ${ampm}`;
};

// --- Reusable Components ---
const Card = ({ title, children, topRightContent = null, className = "" }) => (
    <div className={`glass-card holographic-card rounded-3xl p-6 transition-all duration-300 ease-in-out ${className}`}>
        {(title || topRightContent) && (
            <div className="flex justify-between items-start mb-6 shrink-0">
                {title && <h2 className="text-3xl font-bold text-pink-400 neon-text tracking-tight">{title}</h2>}
                {topRightContent}
            </div>
        )}
        {children}
    </div>
);

const InputField = ({ id, placeholder, value, onChange, className = "" }) => (
    <input 
        type="text" 
        id={id} 
        placeholder={placeholder} 
        value={value} 
        onChange={onChange}
        aria-label={placeholder}
        className={`bg-gray-900/50 text-gray-100 placeholder-gray-400 px-5 py-3 rounded-xl border border-gray-700/50 focus:ring-2 focus:ring-pink-400 outline-none transition-all duration-200 shadow-inner hover:bg-gray-800/70 ${className}`} 
    />
);

const Button = ({ onClick, children, className = "", variant = "primary", disabled = false }) => {
    const baseStyle = "font-semibold py-3 px-6 rounded-xl transition-all duration-300 ease-in-out transform glow-button";
    const primaryStyle = `bg-gradient-to-r from-pink-500 to-purple-500 text-white ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'}`;
    const secondaryStyle = `bg-gradient-to-r from-gray-600 to-gray-700 text-gray-100 ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'}`;
    return (
        <button 
            onClick={onClick} 
            disabled={disabled} 
            aria-disabled={disabled}
            className={`${baseStyle} ${variant === "secondary" ? secondaryStyle : primaryStyle} ${className}`}
        >
            {children}
        </button>
    );
};

// --- Main Components ---
const HeaderControls = ({ onFetch, initialCity }) => {
    const [city, setCity] = useState(initialCity.city);
    const [state, setState] = useState(initialCity.state);
    const [country, setCountry] = useState(initialCity.country);
    const [activePreset, setActivePreset] = useState(initialCity.name);

    const handleFetch = () => {
        if (!city || !state || !country) { 
            alert("City, State, and Country are required."); 
            return; 
        }
        onFetch({ city, state, country });
    };

    const handlePresetClick = (preset) => {
        setCity(preset.city); 
        setState(preset.state); 
        setCountry(preset.country);
        setActivePreset(preset.name);
        onFetch({ city: preset.city, state: preset.state, country: preset.country });
    };

    useEffect(() => {
        setCity(initialCity.city); 
        setState(initialCity.state); 
        setCountry(initialCity.country);
        setActivePreset(initialCity.name);
    }, [initialCity]);

    return (
        <div className="relative mb-12">
            <div className="absolute inset-0 bg-gradient-to-b from-pink-500/20 to-transparent rounded-t-3xl h-48"></div>
            <Card title="AeroSense Dashboard" className="relative z-10 max-w-5xl mx-auto">
                <div className="flex justify-between items-center mb-6">
                    <p className="text-sm text-gray-400">Cybernetic Air Quality Monitoring</p>
                    <div className="flex items-center space-x-3">
                        <span className="text-sm text-pink-400 animate-pulse">
                            <i className="fas fa-satellite mr-1.5"></i>Live
                        </span>
                    </div>
                </div>
                <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-6">
                    <div className="flex flex-col sm:flex-row gap-4 w-full md:w-auto">
                        <InputField 
                            id="cityInput" 
                            placeholder="Enter City" 
                            value={city} 
                            onChange={(e) => { setCity(e.target.value); setActivePreset(null); }} 
                        />
                        <InputField 
                            id="stateInput" 
                            placeholder="Enter State" 
                            value={state} 
                            onChange={(e) => { setState(e.target.value); setActivePreset(null); }} 
                        />
                        <InputField 
                            id="countryInput" 
                            placeholder="Enter Country" 
                            value={country} 
                            onChange={(e) => { setCountry(e.target.value); setActivePreset(null); }} 
                        />
                    </div>
                    <Button onClick={handleFetch} className="w-full md:w-auto">
                        <i className="fas fa-search mr-2"></i>Scan Atmosphere
                    </Button>
                </div>
                <div className="mt-6 pt-6 border-t border-gray-700/30">
                    <h3 className="text-lg font-medium text-gray-300 mb-4">Hotspots (India)</h3>
                    <div className="flex flex-wrap gap-3">
                        {MAJOR_CITIES.map(pCity => (
                            <button 
                                key={pCity.name} 
                                onClick={() => handlePresetClick(pCity)}
                                className={`text-sm px-5 py-2.5 rounded-xl transition-all duration-300 ease-in-out transform glow-button
                                    ${activePreset === pCity.name
                                        ? 'bg-gradient-to-r from-pink-500 to-purple-500 text-white'
                                        : 'bg-gray-900/50 hover:bg-gray-800/70 text-gray-200 hover:text-white'}`}
                                aria-label={`Select ${pCity.name}`}
                            > 
                                {pCity.name} 
                            </button>
                        ))}
                    </div>
                </div>
            </Card>
        </div>
    );
};

const CurrentAQCard = ({ data }) => {
    if (!data) return (
        <Card title="Current Atmosphere" className="h-full flex flex-col min-w-0">
            <p className="text-gray-400 text-center flex-grow flex items-center justify-center text-lg">
                Select a location to scan the atmosphere.
            </p>
        </Card>
    );

    const aqiInfo = getAqiInfo(data.aqi_us);
    return (
        <Card title="Current Atmosphere" className="h-full flex flex-col min-w-0">
            <div className="text-2xl font-semibold text-pink-400 mb-4 shrink-0 neon-text">
                {data.city}{data.state ? `, ${data.state}` : ''}
            </div>
            <div className="text-center my-auto flex-grow flex flex-col justify-center">
                <div className={`relative mx-auto w-40 h-40 rounded-full flex items-center justify-center ${aqiInfo.bgColorClass} glow-aqi-${aqiInfo.glowColor} aqi-circle aqi-circle-bg`}>
                    <div className="absolute inset-0 bg-black/70 rounded-full"></div>
                    <p className="relative text-6xl font-black text-white tracking-tight aqi-text">
                        {data.aqi_us ?? '--'}
                    </p>
                </div>
                <p className={`text-3xl font-semibold ${aqiInfo.colorClass} mt-4`}>{aqiInfo.label}</p>
            </div>
            <div className="grid grid-cols-2 gap-x-6 gap-y-4 text-lg mb-6 shrink-0">
                {[
                    { icon: "fa-wind", label: "Wind", value: data.wind_speed_mps?.toFixed(1), unit: "m/s" },
                    { icon: "fa-tint", label: "Humidity", value: data.humidity_percent, unit: "%" },
                    { icon: "fa-thermometer-half", label: "Temp", value: data.temperature_celsius, unit: "°C" },
                    { icon: "fa-compress-arrows-alt", label: "Pressure", value: data.pressure_hpa || 'N/A', unit: data.pressure_hpa ? "hPa" : "" }
                ].map(item => (
                    <div key={item.label} className="flex items-center">
                        <strong className="text-gray-400">
                            <i className={`fas ${item.icon} mr-2 text-purple-400`}></i>{item.label}:
                        </strong>
                        <span className="ml-2 text-gray-100">{item.value ?? '--'}{item.unit}</span>
                    </div>
                ))}
            </div>
            <div className="mt-auto pt-6 border-t border-gray-700/30 shrink-0">
                <h3 className="text-xl font-semibold text-gray-100 mb-3">Safety Protocol</h3>
                <div className={aqiInfo.recommendationClasses}>
                    <p className="text-base">{aqiInfo.recommendation}</p>
                </div>
            </div>
            <p className="text-sm text-gray-500 mt-5 text-right shrink-0">
                Last Scanned: {formatTimestampToIST(data.timestamp)}
            </p>
        </Card>
    );
};

const PredictionChartComponent = ({ predictionData, title, timeUnit = 'hour' }) => {
    if (!predictionData || predictionData.length === 0) return null;

    const datasets = [
        {
            label: 'Predicted AQI',
            data: predictionData.map(p => p.predicted_aqi_us),
            borderColor: '#F472B6',
            backgroundColor: 'rgba(244, 114, 182, 0.2)',
            fill: true,
            tension: 0.4,
            pointRadius: predictionData.length < 30 ? 4 : 2,
            pointBackgroundColor: '#F472B6',
            pointBorderColor: '#1E293B',
            pointHoverRadius: 6,
            pointHoverBorderWidth: 2,
            pointHoverBorderColor: '#fff'
        }
    ];

    const hasActualData = predictionData.some(p => p.actual_aqi_us !== undefined && p.actual_aqi_us !== null);
    if (hasActualData) {
        datasets.push({
            label: 'Actual AQI',
            data: predictionData.map(p => p.actual_aqi_us ?? null),
            borderColor: '#A78BFA',
            backgroundColor: 'rgba(167, 139, 250, 0.2)',
            fill: false,
            tension: 0.4,
            pointRadius: predictionData.length < 30 ? 4 : 2,
            pointBackgroundColor: '#A78BFA',
            pointBorderColor: '#1E293B',
            pointHoverRadius: 6,
            pointHoverBorderWidth: 2,
            pointHoverBorderColor: '#fff',
            borderDash: [5, 5]
        });
    }

    const chartData = {
        labels: predictionData.map(p => new Date(p.timestamp)),
        datasets
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { 
                type: 'time', 
                time: { 
                    unit: timeUnit, 
                    tooltipFormat: 'MMM d, HH:mm', 
                    displayFormats: { 
                        hour: 'HH:mm', 
                        day: 'MMM d', 
                        week: 'MMM dd', 
                        month: 'MMM yy', 
                        year: 'yyyy' 
                    } 
                }, 
                ticks: { color: '#94A3B8', maxRotation: 0, autoSkipPadding: 20, font: { size: 12 } }, 
                grid: { color: 'rgba(100, 116, 139, 0.15)' } 
            },
            y: { 
                beginAtZero: true, 
                ticks: { color: '#94A3B8', font: { size: 12 } }, 
                grid: { color: 'rgba(100, 116, 139, 0.15)' }, 
                grace: '10%',
                title: { display: true, text: 'AQI (US)', color: '#94A3B8', font: { size: 14 } }
            }
        },
        plugins: { 
            legend: { 
                labels: { 
                    color: '#E2E8F0', 
                    font: { size: 14 },
                    usePointStyle: true,
                    padding: 20
                },
                position: 'top'
            }, 
            tooltip: { 
                mode: 'index', 
                intersect: false, 
                backgroundColor: 'rgba(15, 23, 42, 0.9)', 
                titleFont: { weight: 'bold', size: 14 }, 
                bodyFont: { size: 12 }, 
                bodySpacing: 6, 
                padding: 12, 
                cornerRadius: 8, 
                borderColor: 'rgba(244, 114, 182, 0.5)', 
                borderWidth: 1,
                callbacks: {
                    label: (context) => {
                        const label = context.dataset.label || '';
                        const value = context.raw;
                        return value !== null ? `${label}: ${value.toFixed(1)}` : `${label}: N/A`;
                    }
                }
            },
            title: {
                display: true,
                text: title,
                color: '#E2E8F0',
                font: { size: 16, weight: 'bold' },
                padding: { top: 10, bottom: 20 }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };

    return <Line data={chartData} options={options} />;
};

const PredictionAnalysisCard = ({ currentSearchLocation }) => {
    const [activeTab, setActiveTab] = useState('Hourly');
    const [predictionData, setPredictionData] = useState([]);
    const [message, setMessage] = useState('');
    const [modelName, setModelName] = useState('N/A');
    const [isLoading, setIsLoading] = useState(false);

    const fetchPredictionData = useCallback(async (tab) => {
        if (!currentSearchLocation || !currentSearchLocation.city) {
            setPredictionData([]);
            setMessage(currentSearchLocation?.city ? `Select a forecast type.` : 'Location not selected for prediction.');
            return;
        }
        setIsLoading(true);
        setMessage('');
        setModelName('N/A');
        setPredictionData([]);

        let endpoint = '';
        let params = `city=${encodeURIComponent(currentSearchLocation.city)}`;
        if (currentSearchLocation.state) params += `&state=${encodeURIComponent(currentSearchLocation.state)}`;
        if (currentSearchLocation.country) params += `&country=${encodeURIComponent(currentSearchLocation.country)}`;

        if (tab === 'Hourly') endpoint = `/air-quality/predict?${params}&hours_to_predict=24`;
        else if (tab === 'Daily') endpoint = `/air-quality/predict/daily?${params}&days_to_predict=7`;
        else if (tab === 'Weekly') endpoint = `/air-quality/predict/weekly?${params}&weeks_to_predict=4`;
        else { 
            setIsLoading(false); 
            return; 
        }

        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`);
            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
                throw new Error(err.detail || `Request failed with status ${response.status}`);
            }
            const data = await response.json();
            setPredictionData(data.predictions || []);
            setMessage(data.message || (data.predictions && data.predictions.length > 0 ? 'Forecast loaded.' : 'No prediction data available.'));
            setModelName(data.predictions?.[0]?.forecast_model || (data.message && data.message.includes("Random Fallback")) ? "Random Fallback" : 'N/A');
        } catch (error) {
            console.error(`Error fetching ${tab.toLowerCase()} predictions:`, error);
            setMessage(`Failed to load ${tab.toLowerCase()} predictions: ${error.message}`);
            setPredictionData([]);
        } finally {
            setIsLoading(false);
        }
    }, [currentSearchLocation]);

    useEffect(() => {
        if (currentSearchLocation?.city) {
            fetchPredictionData(activeTab);
        } else {
            setPredictionData([]);
            setMessage('Select a city to see predictions.');
            setIsLoading(false);
        }
    }, [activeTab, fetchPredictionData, currentSearchLocation]);

    const getTimeUnitForChart = () => {
        if (activeTab === 'Hourly') return 'hour';
        if (activeTab === 'Daily') return 'day';
        if (activeTab === 'Weekly') return 'week';
        return 'hour';
    };

    return (
        <Card title="AQI Forecast" className="h-full flex flex-col min-w-0">
            <div className="flex justify-between items-center mb-4 shrink-0">
                <span className="text-sm text-gray-400">
                    Forecast Model: <span className="text-pink-400 font-semibold">{modelName}</span>
                </span>
            </div>
            <div className="mb-6 flex space-x-1 bg-gray-900/50 p-1 rounded-xl shadow-inner">
                {['Hourly', 'Daily', 'Weekly'].map(tab => (
                    <button 
                        key={tab} 
                        onClick={() => setActiveTab(tab)}
                        className={`flex-1 py-2.5 px-4 text-sm font-medium rounded-lg transition-all duration-300 ease-in-out glow-button
                            ${activeTab === tab ? 'bg-gradient-to-r from-pink-500 to-purple-500 text-white' : 'text-gray-300 hover:bg-gray-800/70 hover:text-gray-100'}`}
                        aria-label={`View ${tab} forecast`}
                    >
                        {tab}
                    </button>
                ))}
            </div>
            <div className="flex-grow min-h-0 relative">
                {isLoading ? (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 rounded-lg">
                        <i className="fas fa-spinner fa-spin text-5xl text-pink-400"></i>
                    </div>
                ) : (
                    <>
                        {predictionData.length > 0 ? (
                            <div className="h-[500px]">
                                <PredictionChartComponent 
                                    predictionData={predictionData} 
                                    title={`AQI Forecast (${activeTab})`} 
                                    timeUnit={getTimeUnitForChart()} 
                                />
                            </div>
                        ) : (
                            <div className="h-full flex items-center justify-center">
                                <p className="text-gray-400 text-center text-lg">
                                    {message || `No ${activeTab.toLowerCase()} prediction data available.`}
                                </p>
                            </div>
                        )}
                        {message && predictionData.length > 0 && (
                            <p className="mt-3 text-sm text-gray-400 text-center">
                                {message}
                            </p>
                        )}
                    </>
                )}
            </div>
        </Card>
    );
};

const MapCard = () => {
    const mapRef = useRef(null);
    const mapInstanceRef = useRef(null);
    const [mapData, setMapData] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchMapData = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const requests = MAJOR_CITIES.map(({ city, state, country }) => 
                fetch(`${API_BASE_URL}/air-quality/current/iqair?city=${encodeURIComponent(city)}&state=${encodeURIComponent(state)}&country=${encodeURIComponent(country)}`)
            );
            const responses = await Promise.all(requests);
            const data = await Promise.all(responses.map(async (res, index) => {
                if (!res.ok) {
                    throw new Error(`Failed to fetch data for ${MAJOR_CITIES[index].city}: ${res.statusText}`);
                }
                const json = await res.json();
                return {
                    ...MAJOR_CITIES[index],
                    aqi: json.aqi_us || null,
                    temperature: json.temperature_celsius || null,
                    fetched_at: json.fetched_at || new Date().toISOString()
                };
            }));
            setMapData(data);
            setError(null);
        } catch (err) {
            console.error("Error fetching map data:", err);
            setError("Failed to load air quality map data. Displaying fallback data.");
            setMapData(MAJOR_CITIES.map(city => ({
                ...city,
                aqi: Math.floor(Math.random() * 150) + 20,
                temperature: (Math.random() * 15 + 20).toFixed(1),
                fetched_at: new Date().toISOString()
            })));
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchMapData();
        const interval = setInterval(() => {
            console.log("Polling map data at", new Date().toISOString());
            fetchMapData();
        }, POLLING_INTERVAL);

        return () => clearInterval(interval);
    }, [fetchMapData]);

    useEffect(() => {
        if (!mapRef.current || !window.L) return;

        if (!mapInstanceRef.current) {
            const map = L.map(mapRef.current, { zoomControl: false }).setView([20.5937, 78.9629], 4.8);
            L.control.zoom({ position: 'bottomright' }).addTo(map);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
                maxZoom: 18,
            }).addTo(map);

            mapInstanceRef.current = map;

            const resizeObserver = new ResizeObserver(() => {
                map.invalidateSize();
            });
            resizeObserver.observe(mapRef.current);

            return () => {
                resizeObserver.disconnect();
                map.remove();
                mapInstanceRef.current = null;
            };
        }
    }, []);

    useEffect(() => {
        if (!mapInstanceRef.current || !mapData.length) return;

        const map = mapInstanceRef.current;

        map.eachLayer(layer => {
            if (layer instanceof L.CircleMarker) {
                map.removeLayer(layer);
            }
        });

        mapData.forEach(point => {
            if (isNaN(point.lat) || isNaN(point.lon)) {
                console.warn(`Invalid coordinates for ${point.city}: lat=${point.lat}, lon=${point.lon}`);
                return;
            }

            const aqiInfo = getAqiInfo(point.aqi);
            const colorHex = 
                aqiInfo.label === "Good" ? '#4ADE80' :
                aqiInfo.label === "Moderate" ? '#FACC15' :
                aqiInfo.label === "Unhealthy for Sensitive" ? '#FB923C' :
                aqiInfo.label === "Unhealthy" ? '#F87171' :
                aqiInfo.label === "Very Unhealthy" ? '#C084FC' :
                aqiInfo.label === "Hazardous" ? '#F472B6' :
                '#94A3B8';

            L.circleMarker([point.lat, point.lon], { 
                radius: 9, 
                fillColor: colorHex, 
                color: "#1E293B", 
                weight: 1.5, 
                opacity: 1, 
                fillOpacity: 0.9 
            }).addTo(map).bindPopup(`
                <b>${point.city}</b><br>
                <strong>Current AQI:</strong> ${point.aqi ?? 'N/A'} (${aqiInfo.label})<br>
                <strong>Temperature:</strong> ${point.temperature ?? 'N/A'}°C<br>
                <strong>Updated:</strong> ${formatTimestampToIST(point.fetched_at)}
            `);
        });
    }, [mapData]);

    return (
        <Card 
            title="Air Quality Map" 
            topRightContent={
                <span className="text-sm text-pink-400 flex items-center animate-pulse">
                    <i className="fas fa-broadcast-tower mr-1.5"></i>
                    {error ? 'Fallback Data' : 'Live Data'}
                </span>
            } 
            className="h-full flex flex-col min-w-0"
        >
            <div 
                ref={mapRef} 
                className="bg-gray-900 rounded-xl shadow-inner overflow-hidden flex-grow min-h-0 h-full"
            >
                {isLoading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50">
                        <i className="fas fa-spinner fa-spin text-5xl text-pink-400"></i>
                    </div>
                )}
                {!window.L && <p className="text-gray-400 p-4">Leaflet map library not loaded.</p>}
            </div>
            {error && (
                <p className="text-sm text-red-400 mt-4 text-center shrink-0">{error}</p>
            )}
            <p className="text-sm text-gray-500 mt-4 text-center shrink-0">
                Interactive global air quality visualization
            </p>
        </Card>
    );
};

// --- App Component ---
export default function App() {
    const [currentLocationData, setCurrentLocationData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [currentSearch, setCurrentSearch] = useState(MAJOR_CITIES[0]);

    const fetchDataForLocation = useCallback(async ({ city, state, country }) => {
        setIsLoading(true);
        setError(null);
        try {
            let currentUrl = `${API_BASE_URL}/air-quality/current/iqair?city=${encodeURIComponent(city)}`;
            if (state) currentUrl += `&state=${encodeURIComponent(state)}`;
            if (country) currentUrl += `&country=${encodeURIComponent(country)}`;

            const currentResponse = await fetch(currentUrl);
            if (!currentResponse.ok) {
                const err = await currentResponse.json().catch(() => ({ detail: `HTTP error ${currentResponse.status}` }));
                throw new Error(`Failed to fetch IQAir data: ${err.detail || currentResponse.statusText}`);
            }
            const currentData = await currentResponse.json();
            setCurrentLocationData(currentData);
            setError(null);
        } catch (err) {
            console.error("Fetch error (IQAir data):", err);
            setError(err.message);
            setCurrentLocationData(null);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        if (currentSearch && currentSearch.city) {
            fetchDataForLocation(currentSearch);

            const interval = setInterval(() => {
                console.log("Polling current location data at", new Date().toISOString());
                fetchDataForLocation(currentSearch);
            }, POLLING_INTERVAL);

            return () => clearInterval(interval);
        } else {
            setIsLoading(false);
            setCurrentLocationData(null);
            setError("Please select a city.");
        }
    }, [fetchDataForLocation, currentSearch]);

    return (
        <div className="min-h-screen gradient-bg font-sans transition-colors duration-300 relative overflow-hidden">
            <div className="absolute inset-0 particle-bg z-0"></div>
            <div className="container mx-auto p-4 sm:p-6 lg:p-10 xl:p-12 min-w-[1024px] relative z-10">
                <div className="absolute top-0 left-0 w-full h-64 bg-gradient-to-b from-pink-500/10 to-transparent"></div>
                <HeaderControls onFetch={setCurrentSearch} initialCity={currentSearch} />

                {isLoading && (
                    <div className="fixed inset-0 bg-gray-900/90 flex items-center justify-center z-50">
                        <div className="text-center p-8 glass-card rounded-2xl">
                            <i className="fas fa-spinner fa-spin text-6xl text-pink-400 mb-4"></i>
                            <p className="text-gray-200 text-xl font-semibold">Scanning Atmosphere...</p>
                        </div>
                    </div>
                )}
                {error && !isLoading && (
                    <div className="my-10 p-8 glass-card border-2 border-red-600/50 text-red-300 rounded-2xl text-center">
                        <p className="font-bold text-2xl mb-3">
                            <i className="fas fa-exclamation-triangle mr-2"></i>Error Detected
                        </p>
                        <p className="text-red-200 text-lg">{error}</p>
                        <Button 
                            onClick={() => fetchDataForLocation(currentSearch)} 
                            className="mt-5" 
                            variant="secondary"
                        >
                            <i className="fas fa-sync-alt mr-2"></i>Retry Scan
                        </Button>
                    </div>
                )}

                {!isLoading && !error && (
                    <div className="relative z-10">
                        <div className="grid grid-cols-1 lg:grid-cols-[70%_30%] gap-6 xl:gap-8">
                            <div className="lg:col-start-1">
                                <MapCard />
                            </div>
                            <div className="lg:col-start-2">
                                <CurrentAQCard data={currentLocationData} />
                            </div>
                            <div className="lg:col-span-2">
                                <PredictionAnalysisCard currentSearchLocation={currentSearch} />
                            </div>
                        </div>
                    </div>
                )}
                {!isLoading && !error && !currentLocationData && currentSearch?.city && (
                    <div className="my-10 p-8 glass-card border-2 border-yellow-600/50 text-yellow-300 rounded-2xl text-center">
                        <p className="font-bold text-2xl mb-3">
                            <i className="fas fa-info-circle mr-2"></i>No Data Found
                        </p>
                        <p className="text-yellow-200 text-lg">
                            Unable to retrieve air quality data for {currentSearch.city}. The location may be offline or unsupported.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}