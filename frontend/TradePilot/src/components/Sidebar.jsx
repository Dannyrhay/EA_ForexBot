import { LayoutDashboard, LineChart, Settings, Activity, History, X } from "lucide-react";
import { cn } from "@/lib/utils";

export function Sidebar({ className, activeTab, setActiveTab, isOpen, onClose }) {
    const navItems = [
        { name: "Dashboard", icon: LayoutDashboard, id: "dashboard" },
        { name: "Strategies", icon: Activity, id: "strategies" },
        { name: "Trades", icon: History, id: "trades" },
        { name: "Settings", icon: Settings, id: "settings" },
    ];

    return (
        <div className={cn(
            "pb-12 w-64 border-r bg-card h-screen fixed left-0 top-0 z-50 transition-transform duration-300 ease-in-out md:translate-x-0",
            isOpen ? "translate-x-0" : "-translate-x-full",
            className
        )}>
            <div className="space-y-4 py-4">
                <div className="px-3 py-2">
                    <h2 className="mb-2 px-4 text-lg font-semibold tracking-tight text-primary flex items-center justify-between">
                        TradePilot
                        <button className="md:hidden" onClick={onClose}>
                            <X className="h-5 w-5" />
                        </button>
                    </h2>
                    <div className="space-y-1">
                        {navItems.map((item) => (
                            <button
                                key={item.id}
                                onClick={() => setActiveTab(item.id)}
                                className={cn(
                                    "w-full flex items-center rounded-md px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground transition-colors",
                                    activeTab === item.id ? "bg-accent text-accent-foreground" : "text-muted-foreground"
                                )}
                            >
                                <item.icon className="mr-2 h-4 w-4" />
                                {item.name}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
