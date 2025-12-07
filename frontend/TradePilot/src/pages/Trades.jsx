import React, { useState } from 'react';
import { useQuery } from "@tanstack/react-query";
import { fetchTrades } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Search, Filter, Download, ChevronLeft, ChevronRight, Edit2, XCircle, ArrowUp, ArrowDown } from "lucide-react";

export default function Trades() {
    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("Active");
    const [strategyFilter, setStrategyFilter] = useState("All");

    const { data: trades, isLoading } = useQuery({
        queryKey: ['trades'],
        queryFn: fetchTrades,
        refetchInterval: 5000,
    });

    // Filter logic (client-side for now)
    const filteredTrades = trades?.filter(trade => {
        const matchesSearch = trade.symbol.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesStatus = statusFilter === "All" || trade.status === statusFilter;
        const matchesStrategy = strategyFilter === "All" || trade.strategy === strategyFilter;
        return matchesSearch && matchesStatus && matchesStrategy;
    }) || [];

    return (
        <div className="space-y-6 max-w-6xl mx-auto">
            <div className="flex flex-col gap-4">
                <h2 className="text-3xl font-bold tracking-tight">Advanced Trade Table</h2>

                {/* Toolbar */}
                <Card className="bg-card/50 backdrop-blur-sm border-muted/40">
                    <CardContent className="p-4">
                        <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
                            {/* Search */}
                            <div className="relative w-full md:w-72">
                                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                                <input
                                    type="text"
                                    placeholder="Search Symbol..."
                                    className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                            </div>

                            {/* Filters */}
                            <div className="flex flex-wrap gap-3 w-full md:w-auto items-center">
                                <select
                                    className="bg-background border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                                    value={strategyFilter}
                                    onChange={(e) => setStrategyFilter(e.target.value)}
                                >
                                    <option value="All">All Strategies</option>
                                    <option value="SMC">SMC</option>
                                    <option value="Fibonacci">Fibonacci</option>
                                    <option value="MalaysianSnR">MalaysianSnR</option>
                                </select>

                                <select
                                    className="bg-background border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                                    value={statusFilter}
                                    onChange={(e) => setStatusFilter(e.target.value)}
                                >
                                    <option value="All">All Status</option>
                                    <option value="Active">Active</option>
                                    <option value="Closed">Closed</option>
                                    <option value="Pending">Pending</option>
                                </select>

                                <select className="bg-background border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50">
                                    <option>Last 24 Hours</option>
                                    <option>Last 7 Days</option>
                                    <option>Last 30 Days</option>
                                </select>

                                <button className="flex items-center px-4 py-2 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md text-sm font-medium transition-colors">
                                    <Download className="w-4 h-4 mr-2" /> Export CSV
                                </button>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Table */}
            <Card className="overflow-hidden border-muted/40">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-muted/50 text-muted-foreground uppercase text-xs font-medium">
                            <tr>
                                <th className="px-6 py-4">Symbol</th>
                                <th className="px-6 py-4">Type</th>
                                <th className="px-6 py-4">Size</th>
                                <th className="px-6 py-4">Entry Price</th>
                                <th className="px-6 py-4">Current Price</th>
                                <th className="px-6 py-4">P/L</th>
                                <th className="px-6 py-4 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border/50">
                            {isLoading ? (
                                <tr>
                                    <td colSpan="7" className="px-6 py-8 text-center text-muted-foreground">Loading trades...</td>
                                </tr>
                            ) : filteredTrades.length === 0 ? (
                                <tr>
                                    <td colSpan="7" className="px-6 py-8 text-center text-muted-foreground">No trades found matching your criteria.</td>
                                </tr>
                            ) : (
                                filteredTrades.map((trade, index) => (
                                    <tr key={index} className="hover:bg-muted/30 transition-colors">
                                        <td className="px-6 py-4 font-medium">
                                            <div className="flex items-center gap-3">
                                                {/* Placeholder Icon */}
                                                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold text-primary">
                                                    {trade.symbol.substring(0, 2)}
                                                </div>
                                                <div>
                                                    <div className="text-base">{trade.symbol}</div>
                                                    <div className="text-xs text-muted-foreground">{trade.strategy || 'Unknown'}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2 py-1 rounded text-xs font-bold uppercase ${trade.type === 'BUY'
                                                    ? 'bg-green-500/20 text-green-500 border border-green-500/30'
                                                    : 'bg-red-500/20 text-red-500 border border-red-500/30'
                                                }`}>
                                                {trade.type}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-muted-foreground">{trade.size} Lots</td>
                                        <td className="px-6 py-4 font-mono">{trade.entryPrice}</td>
                                        <td className="px-6 py-4 font-mono">{trade.currentPrice}</td>
                                        <td className="px-6 py-4 font-bold">
                                            <div className={`flex items-center ${trade.pl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                                {trade.pl >= 0 ? '+' : ''}${trade.pl?.toFixed(2)}
                                                {trade.pl >= 0 ? <ArrowUp className="w-3 h-3 ml-1" /> : <ArrowDown className="w-3 h-3 ml-1" />}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <div className="flex items-center justify-end gap-2">
                                                <button className="px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/30 rounded text-xs font-medium transition-colors">
                                                    Close
                                                </button>
                                                <button className="p-1.5 bg-secondary hover:bg-secondary/80 rounded text-muted-foreground transition-colors" title="Edit SL/TP">
                                                    <Edit2 className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Pagination Footer */}
                <div className="flex items-center justify-between px-6 py-4 bg-muted/20 border-t border-border/50">
                    <div className="text-xs text-muted-foreground">
                        Showing {filteredTrades.length} results
                    </div>
                    <div className="flex items-center gap-2">
                        <button className="p-1 rounded hover:bg-accent disabled:opacity-50" disabled>
                            <ChevronLeft className="w-4 h-4" />
                        </button>
                        <span className="text-xs font-medium">Page 1 of 1</span>
                        <button className="p-1 rounded hover:bg-accent disabled:opacity-50" disabled>
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </Card>
        </div>
    );
}
