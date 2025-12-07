import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchStrategyDetails, toggleStrategy } from "@/services/api";
import { Card, CardContent } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { TrendingUp, TrendingDown, Flame, Snowflake, Activity } from "lucide-react";

export default function Strategies() {
    const queryClient = useQueryClient();

    const { data: strategies, isLoading } = useQuery({
        queryKey: ['strategies'],
        queryFn: fetchStrategyDetails,
    });

    const toggleMutation = useMutation({
        mutationFn: ({ id, enabled }) => toggleStrategy(id, enabled),
        onSuccess: () => {
            queryClient.invalidateQueries(['strategies']);
        },
    });

    const handleToggle = (id, currentStatus) => {
        toggleMutation.mutate({ id, enabled: !currentStatus });
    };

    const getStatusBadge = (strategy) => {
        if (!strategy.enabled) return (
            <div className="flex items-center px-3 py-1 rounded-md bg-muted text-muted-foreground text-xs font-bold uppercase tracking-wider border border-border/50">
                OFF
            </div>
        );
        if (strategy.winRate >= 70) return (
            <div className="flex items-center px-3 py-1 rounded-md bg-orange-500/10 text-orange-500 text-xs font-bold uppercase tracking-wider border border-orange-500/20 shadow-[0_0_10px_-3px_rgba(249,115,22,0.5)]">
                <Flame className="w-3 h-3 mr-1 fill-current" /> HOT
            </div>
        );
        if (strategy.winRate <= 45) return (
            <div className="flex items-center px-3 py-1 rounded-md bg-blue-500/10 text-blue-500 text-xs font-bold uppercase tracking-wider border border-blue-500/20 shadow-[0_0_10px_-3px_rgba(59,130,246,0.5)]">
                <Snowflake className="w-3 h-3 mr-1 fill-current" /> COLD
            </div>
        );
        return (
            <div className="flex items-center px-3 py-1 rounded-md bg-green-500/10 text-green-500 text-xs font-bold uppercase tracking-wider border border-green-500/20">
                <Activity className="w-3 h-3 mr-1" /> ACTIVE
            </div>
        );
    };

    if (isLoading) return <div className="p-8 text-center text-muted-foreground">Loading strategies...</div>;

    return (
        <div className="space-y-6 max-w-5xl mx-auto">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Strategy Insights</h2>
                    <p className="text-muted-foreground mt-1">Manage and monitor your active trading algorithms.</p>
                </div>
                <div className="flex gap-2">
                    <button className="p-2 rounded-md hover:bg-accent transition-colors">
                        <Activity className="w-5 h-5 text-muted-foreground" />
                    </button>
                </div>
            </div>

            <div className="overflow-x-auto pb-4">
                <div className="min-w-[800px]">
                    {/* Table Header */}
                    <div className="grid grid-cols-12 gap-4 px-6 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        <div className="col-span-4">Strategy Name</div>
                        <div className="col-span-2 text-center">Win Rate (30D)</div>
                        <div className="col-span-3">Recent Trend</div>
                        <div className="col-span-2 text-center">Status</div>
                        <div className="col-span-1 text-right">Action</div>
                    </div>

                    <div className="space-y-3">
                        {strategies?.map((strategy) => (
                            <Card key={strategy.id} className={`border transition-all duration-200 hover:border-primary/50 ${strategy.enabled ? 'bg-card' : 'bg-muted/30 opacity-75'}`}>
                                <CardContent className="p-0">
                                    <div className="grid grid-cols-12 gap-4 items-center p-4">
                                        {/* Name */}
                                        <div className="col-span-4">
                                            <div className="font-bold text-lg">{strategy.name}</div>
                                            <div className="text-xs text-muted-foreground">{strategy.totalTrades} trades executed</div>
                                        </div>

                                        {/* Win Rate */}
                                        <div className="col-span-2 flex items-center justify-center">
                                            <div className={`text-xl font-bold ${strategy.winRate >= 50 ? 'text-green-500' : 'text-red-500'}`}>
                                                {strategy.winRate}%
                                            </div>
                                            {strategy.winRate >= 50 ?
                                                <TrendingUp className="w-4 h-4 ml-1 text-green-500" /> :
                                                <TrendingDown className="w-4 h-4 ml-1 text-red-500" />
                                            }
                                        </div>

                                        {/* Trend Sparkline */}
                                        <div className="col-span-3 h-[40px]">
                                            {strategy.recentResults && strategy.recentResults.length > 0 ? (
                                                <svg className="w-full h-full overflow-visible" preserveAspectRatio="none">
                                                    <defs>
                                                        <linearGradient id={`gradient-${strategy.id}`} x1="0" y1="0" x2="0" y2="1">
                                                            <stop offset="0%" stopColor={strategy.winRate >= 50 ? "#22c55e" : "#ef4444"} stopOpacity="0.5" />
                                                            <stop offset="100%" stopColor={strategy.winRate >= 50 ? "#22c55e" : "#ef4444"} stopOpacity="0" />
                                                        </linearGradient>
                                                        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                                                            <feGaussianBlur stdDeviation="2" result="blur" />
                                                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                                                        </filter>
                                                    </defs>
                                                    <path
                                                        d={`M0,${40 - (Math.max(0, strategy.recentResults[0] + 1) * 20)} ${strategy.recentResults.map((r, i) => `L${(i / (strategy.recentResults.length - 1)) * 100}%,${40 - (Math.max(0, r + 1) * 20)}`).join(' ')}`}
                                                        fill="none"
                                                        stroke={strategy.winRate >= 50 ? "#22c55e" : "#ef4444"}
                                                        strokeWidth="2"
                                                        strokeLinecap="round"
                                                        strokeLinejoin="round"
                                                        filter="url(#glow)"
                                                    />
                                                    <path
                                                        d={`M0,${40 - (Math.max(0, strategy.recentResults[0] + 1) * 20)} ${strategy.recentResults.map((r, i) => `L${(i / (strategy.recentResults.length - 1)) * 100}%,${40 - (Math.max(0, r + 1) * 20)}`).join(' ')} V40 H0 Z`}
                                                        fill={`url(#gradient-${strategy.id})`}
                                                        stroke="none"
                                                    />
                                                </svg>
                                            ) : (
                                                <div className="h-full w-full flex items-center justify-center text-xs text-muted-foreground border border-dashed rounded opacity-50">
                                                    No Data
                                                </div>
                                            )}
                                        </div>

                                        {/* Status Badge */}
                                        <div className="col-span-2 flex justify-center">
                                            {getStatusBadge(strategy)}
                                        </div>

                                        {/* Action Switch */}
                                        <div className="col-span-1 flex justify-end">
                                            <Switch
                                                checked={strategy.enabled}
                                                onCheckedChange={() => handleToggle(strategy.id, strategy.enabled)}
                                            />
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-8 flex justify-between text-xs text-muted-foreground px-2">
                <div>Last updated: Just now</div>
                <div>Total Active Strategies: {strategies?.filter(s => s.enabled).length || 0}</div>
            </div>
        </div>
    );
}
