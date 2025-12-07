import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowUpRight, DollarSign, Activity, Percent, Play, Square, TrendingUp } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchAccountInfo, fetchSignals, fetchStrategyPerformance } from "@/services/api";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
    const { data: accountInfo } = useQuery({
        queryKey: ['accountInfo'],
        queryFn: fetchAccountInfo,
        refetchInterval: 5000,
    });

    const { data: signals } = useQuery({
        queryKey: ['signals'],
        queryFn: fetchSignals,
        refetchInterval: 2000,
    });

    const { data: strategyPerformance } = useQuery({
        queryKey: ['strategyPerformance'],
        queryFn: fetchStrategyPerformance,
        refetchInterval: 10000,
    });

    const { data: equityData } = useQuery({
        queryKey: ['equityCurve'],
        queryFn: async () => {
            const res = await fetch('/api/equity_curve?limit=50');
            return res.json();
        },
        refetchInterval: 60000,
    });

    const handleStartBot = async () => {
        try { await fetch('/api/control/start_bot', { method: 'POST' }); } catch (e) { console.error(e); }
    };

    const handleStopBot = async () => {
        try { await fetch('/api/control/stop_bot', { method: 'POST' }); } catch (e) { console.error(e); }
    };

    // Helper for circular progress
    const CircleProgress = ({ value, size = 80, strokeWidth = 8 }) => {
        const radius = (size - strokeWidth) / 2;
        const circumference = radius * 2 * Math.PI;
        const offset = circumference - (value / 100) * circumference;

        return (
            <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
                <svg className="transform -rotate-90 w-full h-full">
                    <circle
                        cx={size / 2}
                        cy={size / 2}
                        r={radius}
                        stroke="currentColor"
                        strokeWidth={strokeWidth}
                        fill="transparent"
                        className="text-muted/20"
                    />
                    <circle
                        cx={size / 2}
                        cy={size / 2}
                        r={radius}
                        stroke="currentColor"
                        strokeWidth={strokeWidth}
                        fill="transparent"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        strokeLinecap="round"
                        className="text-green-500 drop-shadow-[0_0_8px_rgba(34,197,94,0.5)]"
                    />
                </svg>
                <div className="absolute text-xl font-bold">{value}%</div>
            </div>
        );
    };

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
                    <p className="text-muted-foreground">Real-time trading overview</p>
                </div>
                <div className="flex items-center space-x-2">
                    <button
                        onClick={handleStartBot}
                        disabled={accountInfo?.bot_status === 'RUNNING'}
                        className={`flex items-center px-4 py-2 rounded-md font-medium text-white transition-all ${accountInfo?.bot_status === 'RUNNING'
                            ? 'bg-green-600/50 cursor-not-allowed'
                            : 'bg-green-600 hover:bg-green-700 shadow-[0_0_15px_-3px_rgba(22,163,74,0.6)]'
                            }`}
                    >
                        <Play className="w-4 h-4 mr-2" /> Start Bot
                    </button>
                    <button
                        onClick={handleStopBot}
                        disabled={accountInfo?.bot_status === 'STOPPED'}
                        className={`flex items-center px-4 py-2 rounded-md font-medium text-white transition-all ${accountInfo?.bot_status === 'STOPPED'
                            ? 'bg-red-600/50 cursor-not-allowed'
                            : 'bg-red-600 hover:bg-red-700 shadow-[0_0_15px_-3px_rgba(220,38,38,0.6)]'
                            }`}
                    >
                        <Square className="w-4 h-4 mr-2 fill-current" /> Stop Bot
                    </button>
                </div>
            </div>

            {/* Top Stats Row */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card className="relative overflow-hidden border-t-4 border-t-blue-500">
                    <div className="absolute top-0 right-0 p-4 opacity-10"><DollarSign className="w-24 h-24" /></div>
                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">BALANCE</CardTitle></CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">${accountInfo?.balance?.toFixed(2) ?? "0.00"}</div>
                        <div className={`flex items-center mt-1 text-xs font-medium ${accountInfo?.weekly_growth >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                            <TrendingUp className="w-3 h-3 mr-1" /> {accountInfo?.weekly_growth ? `${accountInfo.weekly_growth > 0 ? '+' : ''}${accountInfo.weekly_growth}% this week` : '0.0% this week'}
                        </div>
                    </CardContent>
                </Card>
                <Card className="relative overflow-hidden border-t-4 border-t-indigo-500">
                    <div className="absolute top-0 right-0 p-4 opacity-10"><Activity className="w-24 h-24" /></div>
                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">EQUITY</CardTitle></CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">${accountInfo?.equity?.toFixed(2) ?? "0.00"}</div>
                        <div className="flex items-center mt-1 text-muted-foreground text-xs">Floating P/L: <span className={accountInfo?.profit >= 0 ? "text-green-500 ml-1" : "text-red-500 ml-1"}>${accountInfo?.profit?.toFixed(2) ?? "0.00"}</span></div>
                    </CardContent>
                </Card>
                <Card className="relative overflow-hidden border-t-4 border-t-green-500">
                    <div className="absolute top-0 right-0 p-4 opacity-10"><ArrowUpRight className="w-24 h-24" /></div>
                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">SESSION P/L</CardTitle></CardHeader>
                    <CardContent>
                        <div className={`text-3xl font-bold ${(accountInfo?.session_pl || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                            {accountInfo?.session_pl >= 0 ? '+' : ''}${accountInfo?.session_pl?.toFixed(2) ?? "0.00"}
                        </div>
                        <div className="flex items-center mt-1 text-green-500 text-xs font-medium bg-green-500/10 w-fit px-2 py-0.5 rounded">
                            {accountInfo?.session_growth ? `${accountInfo.session_growth > 0 ? '+' : ''}${accountInfo.session_growth}%` : '0.0%'}
                        </div>
                    </CardContent>
                </Card>
                <Card className="flex items-center justify-between pr-6 border-t-4 border-t-emerald-500">
                    <div className="flex-1">
                        <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">WIN RATE</CardTitle></CardHeader>
                        <CardContent><div className="text-sm text-muted-foreground">All Time</div></CardContent>
                    </div>
                    <CircleProgress value={accountInfo?.win_rate || 0} />
                </Card>
            </div>

            {/* Middle Section: Chart & Signals */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                <Card className="col-span-4 flex flex-col">
                    <CardHeader><CardTitle>Equity Curve</CardTitle></CardHeader>
                    <CardContent className="flex-1 min-h-[350px]">
                        <div className="h-full w-full">
                            {equityData && equityData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={equityData}>
                                        <defs>
                                            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                        <XAxis dataKey="time" stroke="#666" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(v) => v.split(' ')[1]} />
                                        <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v}`} domain={['auto', 'auto']} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }} itemStyle={{ color: '#fff' }} />
                                        <Area type="monotone" dataKey="equity" stroke="#22c55e" strokeWidth={3} fillOpacity={1} fill="url(#colorEquity)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex items-center justify-center h-full text-muted-foreground">Loading chart...</div>
                            )}
                        </div>
                    </CardContent>
                </Card>
                <Card className="col-span-3 flex flex-col">
                    <CardHeader><CardTitle>Active Signals</CardTitle></CardHeader>
                    <CardContent className="flex-1">
                        <div className="grid grid-rows-2 grid-flow-col gap-4 overflow-x-auto pb-4 h-[320px] max-w-full">
                            {signals?.length > 0 ? (
                                signals.map((signal, index) => (
                                    <div key={index} className="w-[280px] p-4 border rounded-xl bg-card/50 hover:bg-accent/50 transition-colors border-l-4 border-l-green-500 h-[140px] flex flex-col justify-between">
                                        <div className="flex justify-between items-start mb-2">
                                            <div>
                                                <h4 className="font-bold text-lg">{signal.symbol} <span className={signal.type === 'buy' ? 'text-green-500' : 'text-red-500'}>{signal.type.toUpperCase()}</span></h4>
                                                <div className="flex gap-2 mt-1">
                                                    <span className="text-xs font-bold bg-green-500/20 text-green-500 px-2 py-0.5 rounded">{(signal.strength * 100).toFixed(0)}% Conf</span>
                                                    <span className="text-xs font-bold bg-blue-500/20 text-blue-500 px-2 py-0.5 rounded">Active</span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex flex-col gap-1 text-xs text-muted-foreground mt-2">
                                            <div className="flex justify-between items-center"><span>Entry:</span> <span className="text-foreground font-mono">${signal.entry || '0.00'}</span></div>
                                            <div className="flex justify-between items-center"><span>SL:</span> <span className="text-foreground font-mono">${signal.sl || '0.00'}</span></div>
                                            <div className="flex justify-between items-center"><span>TP:</span> <span className="text-foreground font-mono">${signal.tp || '0.00'}</span></div>
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="w-full text-center text-muted-foreground py-8 col-span-full row-span-2 flex items-center justify-center">No active signals</div>
                            )}
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Bottom Section: Strategy Performance */}
            <Card>
                <CardHeader><CardTitle>Strategy Performance</CardTitle></CardHeader>
                <CardContent>
                    <div className="space-y-6">
                        {strategyPerformance?.map((strategy, index) => (
                            <div key={index} className="flex flex-col sm:grid sm:grid-cols-12 gap-2 sm:gap-4 items-center">
                                {/* Mobile: Name and Stats in one row */}
                                <div className="flex justify-between w-full sm:col-span-2 sm:block">
                                    <div className="font-medium">{strategy.name}</div>
                                    <div className="flex gap-2 text-sm sm:hidden">
                                        <span className="font-bold">{strategy.winRate}% Win</span>
                                        <span className="text-muted-foreground">{strategy.score ? `${strategy.score}/10` : '-'}</span>
                                    </div>
                                </div>

                                {/* Progress Bar */}
                                <div className="w-full sm:col-span-8">
                                    <div className="h-4 bg-blue-900/30 rounded-full overflow-hidden relative">
                                        <div
                                            className="h-full bg-blue-500 rounded-full shadow-[0_0_10px_rgba(59,130,246,0.6)] relative z-10"
                                            style={{ width: `${strategy.winRate}%` }}
                                        />
                                    </div>
                                </div>

                                {/* Desktop: Stats */}
                                <div className="hidden sm:flex col-span-2 justify-end gap-4 text-sm">
                                    <span className="font-bold">{strategy.winRate}% Win</span>
                                    <span className="text-muted-foreground">{strategy.score ? `${strategy.score}/10` : '-'}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
