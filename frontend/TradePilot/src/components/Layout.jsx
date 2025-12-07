import { Sidebar } from "./Sidebar";
import { useState } from "react";
import Dashboard from "@/pages/Dashboard";
import Strategies from "@/pages/Strategies";
import Trades from "@/pages/Trades";
import Settings from "@/pages/Settings";
import { Menu } from "lucide-react";

export default function Layout() {
    const [activeTab, setActiveTab] = useState("dashboard");
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    return (
        <div className="flex min-h-screen bg-background text-foreground font-sans antialiased overflow-x-hidden">
            {/* Mobile Header */}
            <div className="md:hidden fixed top-0 left-0 right-0 z-40 flex items-center justify-between p-4 bg-card border-b shadow-sm">
                <span className="font-bold text-lg text-primary">TradePilot</span>
                <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 rounded-md hover:bg-accent">
                    <Menu className="h-6 w-6" />
                </button>
            </div>

            <Sidebar
                activeTab={activeTab}
                setActiveTab={(tab) => { setActiveTab(tab); setIsSidebarOpen(false); }}
                isOpen={isSidebarOpen}
                onClose={() => setIsSidebarOpen(false)}
            />

            {/* Overlay */}
            {isSidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-30 md:hidden backdrop-blur-sm"
                    onClick={() => setIsSidebarOpen(false)}
                />
            )}

            <main className="flex-1 md:ml-64 p-4 md:p-8 pt-20 md:pt-8 transition-all duration-300">
                {activeTab === "dashboard" && <Dashboard />}
                {activeTab === "strategies" && <Strategies />}
                {activeTab === "trades" && <Trades />}
                {activeTab === "settings" && <Settings />}
            </main>
        </div>
    );
}
