import { useState, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { ChevronDown, ChevronRight, RotateCcw, Save, AlertCircle, Loader2, Eye, EyeOff, Check } from "lucide-react";
import { fetchConfig, updateConfig } from '@/services/api';

export default function Settings() {
    const [expandedSections, setExpandedSections] = useState({
        mt5: true,
        general: false,
        riskManagement: false,
        strategies: false,
        tradingSessions: false
    });

    const [config, setConfig] = useState(null);
    const [originalConfig, setOriginalConfig] = useState(null);
    const [hasChanges, setHasChanges] = useState(false);
    const [lastSaved, setLastSaved] = useState("Never");
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState(null);
    const [saveMessage, setSaveMessage] = useState(null);
    const [showPassword, setShowPassword] = useState(false);

    // Fetch config on mount
    useEffect(() => {
        loadConfig();
    }, []);

    const loadConfig = async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await fetchConfig();
            setConfig(data);
            setOriginalConfig(JSON.parse(JSON.stringify(data)));
            setLoading(false);
        } catch (err) {
            setError('Failed to load configuration. Make sure the bot is running.');
            setLoading(false);
            console.error('Error loading config:', err);
        }
    };

    const toggleSection = (section) => {
        setExpandedSections(prev => ({
            ...prev,
            [section]: !prev[section]
        }));
    };

    const handleInputChange = (path, value) => {
        setHasChanges(true);
        setSaveMessage(null);
        const keys = path.split('.');
        setConfig(prev => {
            const newConfig = JSON.parse(JSON.stringify(prev));
            let current = newConfig;
            for (let i = 0; i < keys.length - 1; i++) {
                if (!current[keys[i]]) current[keys[i]] = {};
                current = current[keys[i]];
            }
            current[keys[keys.length - 1]] = value;
            return newConfig;
        });
    };

    const handleArrayChange = (path, value) => {
        // Convert comma-separated string to array
        const arr = value.split(',').map(s => s.trim()).filter(s => s);
        handleInputChange(path, arr);
    };

    const handleSave = async () => {
        try {
            setSaving(true);
            setSaveMessage(null);
            const result = await updateConfig(config);
            setHasChanges(false);
            setOriginalConfig(JSON.parse(JSON.stringify(config)));
            const now = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
            setLastSaved(now);
            setSaveMessage({ type: 'success', text: result.message || 'Configuration saved successfully!' });
            setSaving(false);
        } catch (err) {
            setSaveMessage({ type: 'error', text: 'Failed to save configuration.' });
            setSaving(false);
            console.error('Failed to save settings:', err);
        }
    };

    const handleReset = () => {
        if (confirm('Are you sure you want to discard all changes?')) {
            setConfig(JSON.parse(JSON.stringify(originalConfig)));
            setHasChanges(false);
            setSaveMessage(null);
        }
    };

    const SectionHeader = ({ title, section, description }) => (
        <button
            onClick={() => toggleSection(section)}
            className="w-full flex items-center justify-between p-4 hover:bg-accent/50 transition-colors rounded-lg"
        >
            <div className="flex flex-col items-start">
                <h3 className="text-lg font-semibold">{title}</h3>
                {description && <p className="text-xs text-muted-foreground">{description}</p>}
            </div>
            {expandedSections[section] ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
        </button>
    );

    const InputField = ({ label, value, onChange, type = "text", step, min, max, suffix, placeholder, description }) => (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 py-3 border-b border-border/50 last:border-0">
            <div className="flex-1">
                <label className="text-sm font-medium">{label}</label>
                {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
            </div>
            <div className="flex items-center gap-2">
                <input
                    type={type}
                    value={value ?? ''}
                    onChange={(e) => onChange(type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
                    step={step}
                    min={min}
                    max={max}
                    placeholder={placeholder}
                    className="w-48 px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
                {suffix && <span className="text-sm text-muted-foreground">{suffix}</span>}
            </div>
        </div>
    );

    const PasswordField = ({ label, value, onChange, description }) => (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 py-3 border-b border-border/50 last:border-0">
            <div className="flex-1">
                <label className="text-sm font-medium">{label}</label>
                {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
            </div>
            <div className="flex items-center gap-2">
                <div className="relative">
                    <input
                        type={showPassword ? "text" : "password"}
                        value={value ?? ''}
                        onChange={(e) => onChange(e.target.value)}
                        className="w-48 px-3 py-2 pr-10 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                    />
                    <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                </div>
            </div>
        </div>
    );

    const SwitchField = ({ label, checked, onChange, description }) => (
        <div className="flex items-center justify-between py-3 border-b border-border/50 last:border-0">
            <div className="flex-1">
                <label className="text-sm font-medium">{label}</label>
                {description && <p className="text-xs text-muted-foreground mt-1">{description}</p>}
            </div>
            <Switch checked={checked ?? false} onCheckedChange={onChange} />
        </div>
    );

    const SelectField = ({ label, value, onChange, options, description }) => (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 py-3 border-b border-border/50 last:border-0">
            <div className="flex-1">
                <label className="text-sm font-medium">{label}</label>
                {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
            </div>
            <select
                value={value ?? ''}
                onChange={(e) => onChange(e.target.value)}
                className="w-48 px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
                {options.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
            </select>
        </div>
    );

    const TagInput = ({ label, value = [], onChange, description, placeholder }) => (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 py-3 border-b border-border/50 last:border-0">
            <div className="flex-1">
                <label className="text-sm font-medium">{label}</label>
                {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
            </div>
            <input
                type="text"
                value={Array.isArray(value) ? value.join(', ') : value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                className="w-64 px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
        </div>
    );

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <div className="flex flex-col items-center gap-4">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                    <p className="text-muted-foreground">Loading configuration...</p>
                </div>
            </div>
        );
    }

    if (error && !config) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <div className="flex flex-col items-center gap-4 max-w-md text-center">
                    <AlertCircle className="w-12 h-12 text-destructive" />
                    <h3 className="text-xl font-semibold">Configuration Error</h3>
                    <p className="text-muted-foreground">{error}</p>
                    <button
                        onClick={loadConfig}
                        className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    const allStrategies = ['SMC', 'LiquiditySweep', 'Fibonacci', 'ADX', 'MalaysianSnR'];

    return (
        <div className="space-y-6 max-w-5xl mx-auto">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
                    <p className="text-muted-foreground">Configure your trading bot parameters</p>
                </div>
            </div>

            {/* Messages */}
            {hasChanges && (
                <div className="flex items-center gap-3 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                    <AlertCircle className="w-5 h-5 text-yellow-500" />
                    <span className="text-sm font-medium text-yellow-500">You have unsaved changes</span>
                </div>
            )}

            {saveMessage && (
                <div className={`flex items-center gap-3 p-4 rounded-lg ${saveMessage.type === 'success'
                        ? 'bg-green-500/10 border border-green-500/30'
                        : 'bg-red-500/10 border border-red-500/30'
                    }`}>
                    {saveMessage.type === 'success' ? (
                        <Check className="w-5 h-5 text-green-500" />
                    ) : (
                        <AlertCircle className="w-5 h-5 text-red-500" />
                    )}
                    <span className={`text-sm font-medium ${saveMessage.type === 'success' ? 'text-green-500' : 'text-red-500'
                        }`}>{saveMessage.text}</span>
                </div>
            )}

            {/* MT5 Credentials */}
            <Card className="overflow-hidden">
                <SectionHeader title="MT5 Connection" section="mt5" description="MetaTrader 5 account credentials" />
                {expandedSections.mt5 && (
                    <CardContent className="space-y-1 pt-0">
                        <InputField
                            label="Login ID"
                            value={config?.mt5_credentials?.login}
                            onChange={(v) => handleInputChange('mt5_credentials.login', parseInt(v) || 0)}
                            type="number"
                            description="Your MT5 account number"
                        />
                        <PasswordField
                            label="Password"
                            value={config?.mt5_credentials?.password}
                            onChange={(v) => handleInputChange('mt5_credentials.password', v)}
                            description="Your MT5 account password"
                        />
                        <InputField
                            label="Server"
                            value={config?.mt5_credentials?.server}
                            onChange={(v) => handleInputChange('mt5_credentials.server', v)}
                            placeholder="e.g., Exness-MT5Real"
                            description="MT5 broker server name"
                        />
                        <InputField
                            label="Terminal Path"
                            value={config?.mt5_credentials?.mt5_terminal_path}
                            onChange={(v) => handleInputChange('mt5_credentials.mt5_terminal_path', v)}
                            placeholder="Leave empty for default"
                            description="Path to terminal64.exe (optional)"
                        />
                    </CardContent>
                )}
            </Card>

            {/* General Settings */}
            <Card className="overflow-hidden">
                <SectionHeader title="General Settings" section="general" description="Trading symbols, timeframes, and bot parameters" />
                {expandedSections.general && (
                    <CardContent className="space-y-1 pt-0">
                        <TagInput
                            label="Symbols"
                            value={config?.symbols}
                            onChange={(v) => handleArrayChange('symbols', v)}
                            description="Comma-separated list of trading symbols"
                            placeholder="BTCUSDm, XAUUSDm"
                        />
                        <TagInput
                            label="Timeframes"
                            value={config?.timeframes}
                            onChange={(v) => handleArrayChange('timeframes', v)}
                            description="Comma-separated list of timeframes"
                            placeholder="M5, M15"
                        />
                        <InputField
                            label="Max Trades Per Symbol"
                            value={config?.max_trades_per_symbol}
                            onChange={(v) => handleInputChange('max_trades_per_symbol', v)}
                            type="number"
                            min="1"
                            max="20"
                        />
                        <InputField
                            label="Cooldown Period"
                            value={config?.cooldown_period_minutes}
                            onChange={(v) => handleInputChange('cooldown_period_minutes', v)}
                            type="number"
                            min="0"
                            suffix="minutes"
                            description="Wait time between trades on the same symbol"
                        />
                        <InputField
                            label="Monitoring Interval"
                            value={config?.monitoring_interval_seconds}
                            onChange={(v) => handleInputChange('monitoring_interval_seconds', v)}
                            type="number"
                            min="1"
                            suffix="seconds"
                        />
                    </CardContent>
                )}
            </Card>

            {/* Risk Management */}
            <Card className="overflow-hidden">
                <SectionHeader title="Risk Management" section="riskManagement" description="Risk per trade, stop loss, and position sizing" />
                {expandedSections.riskManagement && (
                    <CardContent className="space-y-1 pt-0">
                        <InputField
                            label="Risk Per Trade"
                            value={(config?.risk_percent_per_trade ?? 0) * 100}
                            onChange={(v) => handleInputChange('risk_percent_per_trade', v / 100)}
                            type="number"
                            step="0.5"
                            min="0.1"
                            max="10"
                            suffix="%"
                            description="Percentage of account to risk per trade"
                        />
                        <SelectField
                            label="SL/TP Method"
                            value={config?.risk_management?.method}
                            onChange={(v) => handleInputChange('risk_management.method', v)}
                            options={[
                                { value: 'atr', label: 'ATR-Based' },
                                { value: 'percentage', label: 'Percentage' },
                                { value: 'fixed_pips', label: 'Fixed Pips' }
                            ]}
                            description="How to calculate Stop Loss and Take Profit"
                        />
                        <InputField
                            label="SL ATR Multiplier"
                            value={config?.risk_management?.atr_params?.sl_multiplier}
                            onChange={(v) => handleInputChange('risk_management.atr_params.sl_multiplier', v)}
                            type="number"
                            step="0.1"
                            min="0.5"
                            max="5"
                        />
                        <InputField
                            label="TP Risk:Reward Ratio"
                            value={config?.risk_management?.atr_params?.tp_risk_reward_ratio}
                            onChange={(v) => handleInputChange('risk_management.atr_params.tp_risk_reward_ratio', v)}
                            type="number"
                            step="0.5"
                            min="1"
                            max="10"
                        />
                        <div className="pt-4 pb-2">
                            <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Portfolio Risk Limits</h4>
                        </div>
                        <SwitchField
                            label="Enable Portfolio Risk"
                            checked={config?.portfolio_risk?.enabled}
                            onChange={(v) => handleInputChange('portfolio_risk.enabled', v)}
                            description="Enable portfolio-level risk management"
                        />
                        <InputField
                            label="Max Daily Drawdown"
                            value={config?.portfolio_risk?.max_daily_drawdown_percent}
                            onChange={(v) => handleInputChange('portfolio_risk.max_daily_drawdown_percent', v)}
                            type="number"
                            step="1"
                            min="1"
                            max="50"
                            suffix="%"
                        />
                        <InputField
                            label="Max Portfolio Risk"
                            value={config?.portfolio_risk?.max_portfolio_risk_percent}
                            onChange={(v) => handleInputChange('portfolio_risk.max_portfolio_risk_percent', v)}
                            type="number"
                            step="5"
                            min="5"
                            max="100"
                            suffix="%"
                        />
                    </CardContent>
                )}
            </Card>

            {/* Strategy Configuration */}
            <Card className="overflow-hidden">
                <SectionHeader title="Strategy Configuration" section="strategies" description="Enable/disable strategies and tune parameters" />
                {expandedSections.strategies && (
                    <CardContent className="space-y-6 pt-0">
                        {/* Active Strategies */}
                        <div className="border border-border/50 rounded-lg p-4 space-y-3">
                            <h4 className="font-semibold text-base">Active Strategies</h4>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                {allStrategies.map(strategy => (
                                    <label key={strategy} className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={config?.active_strategies?.includes(strategy) ?? false}
                                            onChange={(e) => {
                                                const current = config?.active_strategies ?? [];
                                                if (e.target.checked) {
                                                    handleInputChange('active_strategies', [...current, strategy]);
                                                } else {
                                                    handleInputChange('active_strategies', current.filter(s => s !== strategy));
                                                }
                                            }}
                                            className="w-4 h-4 rounded border-input"
                                        />
                                        <span className="text-sm">{strategy}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        {/* SMC Strategy */}
                        <div className="border border-border/50 rounded-lg p-4 space-y-3">
                            <h4 className="font-semibold text-base">SMC Settings</h4>
                            <InputField
                                label="Swing Lookback"
                                value={config?.smc_swing_lookback}
                                onChange={(v) => handleInputChange('smc_swing_lookback', v)}
                                type="number"
                                min="5"
                                max="100"
                            />
                            <InputField
                                label="FVG Threshold"
                                value={config?.smc_fvg_threshold}
                                onChange={(v) => handleInputChange('smc_fvg_threshold', v)}
                                type="number"
                                step="0.0001"
                            />
                            <InputField
                                label="Liquidity Tolerance"
                                value={config?.smc_liquidity_tolerance}
                                onChange={(v) => handleInputChange('smc_liquidity_tolerance', v)}
                                type="number"
                                step="0.0001"
                            />
                            <SelectField
                                label="Higher Timeframe"
                                value={config?.smc_higher_timeframe}
                                onChange={(v) => handleInputChange('smc_higher_timeframe', v)}
                                options={[
                                    { value: 'M15', label: 'M15' },
                                    { value: 'M30', label: 'M30' },
                                    { value: 'H1', label: 'H1' },
                                    { value: 'H4', label: 'H4' },
                                    { value: 'D1', label: 'D1' }
                                ]}
                            />
                        </div>

                        {/* Liquidity Sweep */}
                        <div className="border border-border/50 rounded-lg p-4 space-y-3">
                            <h4 className="font-semibold text-base">Liquidity Sweep Settings</h4>
                            <InputField
                                label="Lookback Period"
                                value={config?.liquidity_sweep_params?.lookback_period}
                                onChange={(v) => handleInputChange('liquidity_sweep_params.lookback_period', v)}
                                type="number"
                                min="5"
                                max="50"
                            />
                            <InputField
                                label="EQ Level Tolerance"
                                value={config?.liquidity_sweep_params?.eq_level_tolerance}
                                onChange={(v) => handleInputChange('liquidity_sweep_params.eq_level_tolerance', v)}
                                type="number"
                                step="0.0001"
                            />
                            <SwitchField
                                label="Enable FVG"
                                checked={config?.liquidity_sweep_params?.enable_fvg}
                                onChange={(v) => handleInputChange('liquidity_sweep_params.enable_fvg', v)}
                            />
                            <SwitchField
                                label="Enable MSS Confirmation"
                                checked={config?.liquidity_sweep_params?.enable_mss_confirmation}
                                onChange={(v) => handleInputChange('liquidity_sweep_params.enable_mss_confirmation', v)}
                            />
                        </div>

                        {/* Fibonacci */}
                        <div className="border border-border/50 rounded-lg p-4 space-y-3">
                            <h4 className="font-semibold text-base">Fibonacci Settings</h4>
                            <InputField
                                label="Swing Lookback"
                                value={config?.fibonacci_golden_zone?.swing_lookback}
                                onChange={(v) => handleInputChange('fibonacci_golden_zone.swing_lookback', v)}
                                type="number"
                                min="10"
                                max="200"
                            />
                            <InputField
                                label="Trend EMA Period"
                                value={config?.fibonacci_golden_zone?.trend_ema_period}
                                onChange={(v) => handleInputChange('fibonacci_golden_zone.trend_ema_period', v)}
                                type="number"
                                min="10"
                                max="200"
                            />
                            <InputField
                                label="Signal Strength"
                                value={config?.fibonacci_golden_zone?.signal_strength}
                                onChange={(v) => handleInputChange('fibonacci_golden_zone.signal_strength', v)}
                                type="number"
                                step="0.1"
                                min="0.1"
                                max="1"
                            />
                        </div>

                        {/* ADX */}
                        <div className="border border-border/50 rounded-lg p-4 space-y-3">
                            <h4 className="font-semibold text-base">ADX Filter Settings</h4>
                            <SwitchField
                                label="Enable ADX Filter"
                                checked={config?.adx_signal_filter?.enabled}
                                onChange={(v) => handleInputChange('adx_signal_filter.enabled', v)}
                            />
                            <InputField
                                label="ADX Period"
                                value={config?.adx_period}
                                onChange={(v) => handleInputChange('adx_period', v)}
                                type="number"
                                min="5"
                                max="50"
                            />
                            <InputField
                                label="Min ADX for Entry"
                                value={config?.adx_signal_filter?.min_adx_for_entry}
                                onChange={(v) => handleInputChange('adx_signal_filter.min_adx_for_entry', v)}
                                type="number"
                                min="5"
                                max="50"
                            />
                            <InputField
                                label="ADX Threshold"
                                value={config?.adx_threshold}
                                onChange={(v) => handleInputChange('adx_threshold', v)}
                                type="number"
                                min="10"
                                max="50"
                            />
                        </div>
                    </CardContent>
                )}
            </Card>

            {/* Trading Sessions */}
            <Card className="overflow-hidden">
                <SectionHeader title="Trading Sessions" section="tradingSessions" description="Session times and trading hours" />
                {expandedSections.tradingSessions && (
                    <CardContent className="space-y-1 pt-0">
                        <SwitchField
                            label="Enable Session Filter"
                            checked={config?.trading_sessions?.enabled}
                            onChange={(v) => handleInputChange('trading_sessions.enabled', v)}
                            description="Only trade during specified sessions"
                        />
                        <div className="pt-4 pb-2">
                            <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Session Times (UTC)</h4>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {['asian', 'london', 'ny'].map(session => (
                                <div key={session} className="border border-border/30 rounded-lg p-3 space-y-2">
                                    <h5 className="font-medium capitalize">{session === 'ny' ? 'New York' : session}</h5>
                                    <div className="flex gap-2 text-sm">
                                        <input
                                            type="time"
                                            value={config?.trading_sessions?.sessions?.[session]?.start ?? '00:00'}
                                            onChange={(e) => handleInputChange(`trading_sessions.sessions.${session}.start`, e.target.value)}
                                            className="flex-1 px-2 py-1 bg-background border border-input rounded text-xs"
                                        />
                                        <span className="text-muted-foreground">to</span>
                                        <input
                                            type="time"
                                            value={config?.trading_sessions?.sessions?.[session]?.end ?? '00:00'}
                                            onChange={(e) => handleInputChange(`trading_sessions.sessions.${session}.end`, e.target.value)}
                                            className="flex-1 px-2 py-1 bg-background border border-input rounded text-xs"
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                )}
            </Card>

            {/* Action Buttons */}
            <Card className="bg-card/50 backdrop-blur-sm border-muted/40">
                <CardContent className="p-4">
                    <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                        <div className="text-sm text-muted-foreground">
                            Last saved: {lastSaved}
                        </div>
                        <div className="flex gap-3">
                            <button
                                onClick={handleReset}
                                disabled={!hasChanges}
                                className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${hasChanges
                                        ? 'bg-secondary hover:bg-secondary/80 text-secondary-foreground'
                                        : 'bg-secondary/50 text-secondary-foreground/50 cursor-not-allowed'
                                    }`}
                            >
                                <RotateCcw className="w-4 h-4 mr-2" /> Discard Changes
                            </button>
                            <button
                                onClick={handleSave}
                                disabled={!hasChanges || saving}
                                className={`flex items-center px-6 py-2 rounded-md text-sm font-medium transition-all ${hasChanges && !saving
                                    ? 'bg-primary hover:bg-primary/90 text-primary-foreground shadow-[0_0_15px_-3px_rgba(59,130,246,0.6)]'
                                    : 'bg-primary/50 text-primary-foreground/50 cursor-not-allowed'
                                    }`}
                            >
                                {saving ? (
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                ) : (
                                    <Save className="w-4 h-4 mr-2" />
                                )}
                                {saving ? 'Saving...' : 'Save Changes'}
                            </button>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
