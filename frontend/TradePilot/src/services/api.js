import axios from 'axios';

const api = axios.create({
    baseURL: '/api', // Proxy handles the target
    headers: {
        'Content-Type': 'application/json',
    },
});

export const fetchAccountInfo = async () => {
    const response = await api.get('/account_info');
    return response.data;
};

export const fetchTrades = async () => {
    const response = await api.get('/trades');
    return response.data;
};

export const fetchSignals = async () => {
    const response = await api.get('/signals');
    return response.data;
};

export const fetchStrategyPerformance = async () => {
    const response = await api.get('/strategy_performance');
    return response.data;
};

export const fetchStrategies = async () => {
    const response = await api.get('/strategies');
    return response.data;
};

export const fetchStrategyDetails = async () => {
    const response = await api.get('/strategies/details');
    return response.data;
};

export const toggleStrategy = async (strategyId, enabled) => {
    const response = await api.post(`/strategies/${strategyId}/toggle`, { enabled });
    return response.data;
};

export const fetchConfig = async () => {
    const response = await api.get('/config');
    return response.data;
};

export const updateConfig = async (config) => {
    const response = await api.post('/config/update', config);
    return response.data;
};



export default api;
