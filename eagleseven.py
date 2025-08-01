import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import random
import pytz

# Configure page
st.set_page_config(
    page_title="Eagle Seven LLC Algorithmic Trading Simulator",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Eagle Seven branding
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d5c3d 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #f0f2f6;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1a472a;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        color: #1a472a;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1rem;
        text-align: center;
        background: #f8f9fa;
        border-radius: 5px;
        font-style: italic;
        color: #6c757d;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 0.25rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
    }
    
    .algo-box {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        color: #004085;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #007bff;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Professional Header for Eagle Seven
st.markdown("""
<div class="main-header">
    <h1>🦅 EAGLE SEVEN, LLC</h1>
    <p>Algorithmic Trading Operations Simulator | Built by Muchiri Kahwai</p>
    <p style="font-size: 1rem; margin-top: 1rem;">
        <strong>Demonstrating:</strong> Algorithmic Trading • Global Markets • Position Reconciliation • Post-Trade Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Description and Disclaimer Section
st.markdown("### 📋 About This Application")

st.info("""
**Algorithmic Trading Operations Simulator** - Demonstrating quantitative finance and operational tools development 
relevant to Eagle Seven, LLC's **Junior Trader** position focused on algorithmic trading strategies and global market operations.
""")

st.markdown("**Key Features:**")
st.markdown("""
• **Algorithmic Strategy Execution:** Automated trading strategies with real-time monitoring  
• **Global Markets Coverage:** Asia/European session trading with timezone management  
• **Position Reconciliation:** Start-of-day and end-of-day position tracking  
• **Post-Trade Analysis:** Performance analytics and strategy evaluation  
• **PnL Reporting:** Comprehensive profit & loss tracking and reporting  
• **Compliance Monitoring:** Risk limits and regulatory compliance tools  
• **Strategy Research:** Tools for quantitative strategy development and backtesting  
""")

st.warning("""
**⚠️ Disclaimer:** This application is **not affiliated with or property of Eagle Seven, LLC**. 
It was independently developed by Muchiri Kahwai to showcase quantitative finance and programming skills 
for consideration in the **Junior Trader** role. All market data is simulated for demonstration purposes.
""")

st.caption("*Built with Python, Streamlit, NumPy, Pandas, and Plotly • Operational tools focus for Eagle Seven, LLC*")

st.markdown("---")

# Global Market Times
def get_market_times():
    utc_now = datetime.now(pytz.UTC)
    
    # Market sessions
    markets = {
        'Tokyo': {'tz': 'Asia/Tokyo', 'open': 9, 'close': 15},
        'Hong Kong': {'tz': 'Asia/Hong_Kong', 'open': 9, 'close': 16},
        'London': {'tz': 'Europe/London', 'open': 8, 'close': 16},
        'New York': {'tz': 'America/New_York', 'open': 9, 'close': 16}
    }
    
    market_status = {}
    for market, info in markets.items():
        local_time = utc_now.astimezone(pytz.timezone(info['tz']))
        current_hour = local_time.hour
        
        if info['open'] <= current_hour < info['close']:
            status = "🟢 OPEN"
        else:
            status = "🔴 CLOSED"
        
        market_status[market] = {
            'status': status,
            'local_time': local_time.strftime("%H:%M"),
            'hour': current_hour
        }
    
    return market_status

# Algorithmic Strategy Functions
def generate_signal(strategy_type, price_data, lookback=20):
    """Generate trading signals based on strategy type"""
    if len(price_data) < lookback:
        return 0
    
    current_price = price_data[-1]
    
    if strategy_type == "Mean Reversion":
        # Simple mean reversion
        mean_price = np.mean(price_data[-lookback:])
        std_price = np.std(price_data[-lookback:])
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        
        if z_score > 1.5:
            return -1  # Sell signal
        elif z_score < -1.5:
            return 1   # Buy signal
        else:
            return 0   # No signal
    
    elif strategy_type == "Momentum":
        # Simple momentum strategy
        if len(price_data) >= 10:
            short_ma = np.mean(price_data[-5:])
            long_ma = np.mean(price_data[-10:])
            
            if short_ma > long_ma * 1.01:
                return 1   # Buy signal
            elif short_ma < long_ma * 0.99:
                return -1  # Sell signal
        return 0
    
    elif strategy_type == "Statistical Arbitrage":
        # Pairs trading signal (simplified)
        if len(price_data) >= lookback:
            ratio = price_data[-1] / price_data[-lookback]
            if ratio > 1.05:
                return -1
            elif ratio < 0.95:
                return 1
        return 0
    
    return 0

# Initialize session state
if 'algo_positions' not in st.session_state:
    st.session_state.algo_positions = []
if 'algo_trades' not in st.session_state:
    st.session_state.algo_trades = []
if 'price_history' not in st.session_state:
    st.session_state.price_history = {'SPY': [450 + np.random.normal(0, 2) for _ in range(50)]}
if 'pnl_history' not in st.session_state:
    st.session_state.pnl_history = []
if 'strategy_performance' not in st.session_state:
    st.session_state.strategy_performance = {}

# Enhanced Sidebar for Algorithmic Trading
st.sidebar.markdown('<p class="sidebar-header">🌍 Global Markets</p>', unsafe_allow_html=True)

# Global Market Status
market_times = get_market_times()
for market, info in market_times.items():
    st.sidebar.markdown(f"**{market}:** {info['status']} ({info['local_time']})")

st.sidebar.markdown("---")

# Algorithm Parameters
st.sidebar.markdown('<p class="sidebar-header">⚙️ Algorithm Parameters</p>', unsafe_allow_html=True)

algo_strategy = st.sidebar.selectbox("Trading Strategy", 
    ["Mean Reversion", "Momentum", "Statistical Arbitrage", "Market Making"])

position_size = st.sidebar.number_input("Position Size ($)", value=100000, min_value=10000, max_value=1000000, step=10000)
risk_limit = st.sidebar.number_input("Risk Limit (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.5) / 100
execution_delay = st.sidebar.slider("Execution Latency (ms)", 1, 100, 25)

st.sidebar.markdown("---")

# Market Data Parameters
st.sidebar.markdown('<p class="sidebar-header">📊 Market Data</p>', unsafe_allow_html=True)
base_price = st.sidebar.number_input("SPY Base Price", value=450.0, min_value=300.0, max_value=600.0, step=1.0)
volatility = st.sidebar.number_input("Market Volatility (%)", value=15.0, min_value=5.0, max_value=50.0, step=1.0) / 100

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🤖 Algorithm Control", "📊 Position Dashboard", "📈 Post-Trade Analysis", 
    "🔄 Reconciliation", "📋 PnL Reporting", "🔬 Strategy Research"
])

with tab1:
    st.markdown("### 🤖 Algorithmic Trading Control Center")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Real-time Market Data and Signals
        st.markdown("#### 📈 Market Data & Algorithm Signals")
        
        # Simulate real-time price updates
        if st.button("🔄 Update Market Data", key="update_market"):
            new_price = st.session_state.price_history['SPY'][-1] + np.random.normal(0, volatility * base_price / 100)
            st.session_state.price_history['SPY'].append(new_price)
            if len(st.session_state.price_history['SPY']) > 100:
                st.session_state.price_history['SPY'] = st.session_state.price_history['SPY'][-100:]
        
        # Current market data
        current_price = st.session_state.price_history['SPY'][-1]
        price_change = current_price - st.session_state.price_history['SPY'][-2] if len(st.session_state.price_history['SPY']) > 1 else 0
        
        col_price1, col_price2, col_price3 = st.columns(3)
        with col_price1:
            st.metric("SPY Price", f"${current_price:.2f}", delta=f"{price_change:.2f}")
        with col_price2:
            signal = generate_signal(algo_strategy, st.session_state.price_history['SPY'])
            signal_text = "🟢 BUY" if signal > 0 else "🔴 SELL" if signal < 0 else "⚪ HOLD"
            st.metric("Algorithm Signal", signal_text)
        with col_price3:
            active_algos = len([t for t in st.session_state.algo_trades if t.get('status') == 'active'])
            st.metric("Active Algorithms", active_algos)
        
        # Price chart with signals
        fig = go.Figure()
        
        # Price line
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(len(st.session_state.price_history['SPY']))][::-1]
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=st.session_state.price_history['SPY'],
            mode='lines',
            name='SPY Price',
            line=dict(color='#1a472a', width=2)
        ))
        
        # Signal markers
        if signal != 0:
            fig.add_trace(go.Scatter(
                x=[timestamps[-1]],
                y=[current_price],
                mode='markers',
                name='Signal',
                marker=dict(
                    color='green' if signal > 0 else 'red',
                    size=15,
                    symbol='triangle-up' if signal > 0 else 'triangle-down'
                )
            ))
        
        fig.update_layout(
            title="Real-time Price Feed with Algorithm Signals",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Algorithm Control Panel
        st.markdown("#### ⚡ Algorithm Execution")
        
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            # Strategy selection and parameters
            st.markdown(f"**Strategy:** {algo_strategy}")
            st.markdown(f"**Position Size:** ${position_size:,}")
            st.markdown(f"**Risk Limit:** {risk_limit*100:.1f}%")
            
            # Auto-execution toggle
            auto_execute = st.checkbox("🤖 Auto-Execute Signals", value=False)
            
            if auto_execute and signal != 0:
                if st.button("🚀 Execute Algorithm Trade", type="primary", use_container_width=True):
                    # Simulate execution delay
                    with st.spinner(f'Executing algorithm... ({execution_delay}ms latency)'):
                        time.sleep(execution_delay / 1000)
                    
                    # Record algorithmic trade
                    trade = {
                        'timestamp': datetime.now(),
                        'strategy': algo_strategy,
                        'symbol': 'SPY',
                        'side': 'BUY' if signal > 0 else 'SELL',
                        'quantity': int(position_size / current_price),
                        'price': current_price,
                        'notional': position_size,
                        'signal_strength': abs(signal),
                        'status': 'executed',
                        'latency_ms': execution_delay
                    }
                    
                    st.session_state.algo_trades.append(trade)
                    
                    st.markdown(f"""
                    <div class="algo-box">
                        ✅ <strong>Algorithm Executed!</strong><br>
                        {trade['strategy']}: {trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}<br>
                        Notional: ${trade['notional']:.0f} | Latency: {trade['latency_ms']}ms
                    </div>
                    """, unsafe_allow_html=True)
            
            elif not auto_execute:
                if st.button("📊 Manual Execute", use_container_width=True):
                    st.info("Manual execution mode - select parameters above")
            
            else:
                st.info("⏳ Waiting for algorithm signal...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Algorithm Performance Summary
        st.markdown("#### 📊 Algorithm Performance")
        
        if st.session_state.algo_trades:
            total_trades = len(st.session_state.algo_trades)
            total_notional = sum([t['notional'] for t in st.session_state.algo_trades])
            avg_latency = np.mean([t['latency_ms'] for t in st.session_state.algo_trades])
            
            st.metric("Total Algo Trades", total_trades)
            st.metric("Total Notional", f"${total_notional:,.0f}")
            st.metric("Avg Execution Latency", f"{avg_latency:.1f}ms")
            
            # Strategy breakdown
            strategy_counts = {}
            for trade in st.session_state.algo_trades:
                strategy = trade['strategy']
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            st.markdown("**Strategy Distribution:**")
            for strategy, count in strategy_counts.items():
                st.markdown(f"• {strategy}: {count} trades")
        
        else:
            st.info("No algorithm trades executed yet")

with tab2:
    st.markdown("### 📊 Position Monitoring Dashboard")
    
    if st.session_state.algo_trades:
        # Calculate current positions
        positions = {}
        
        for trade in st.session_state.algo_trades:
            symbol = trade['symbol']
            if symbol not in positions:
                positions[symbol] = {
                    'quantity': 0,
                    'notional': 0,
                    'avg_price': 0,
                    'strategies': set()
                }
            
            quantity = trade['quantity'] if trade['side'] == 'BUY' else -trade['quantity']
            positions[symbol]['quantity'] += quantity
            positions[symbol]['notional'] += trade['notional'] if trade['side'] == 'BUY' else -trade['notional']
            positions[symbol]['strategies'].add(trade['strategy'])
        
        # Calculate average prices and current P&L
        for symbol, pos in positions.items():
            if pos['quantity'] != 0:
                pos['avg_price'] = abs(pos['notional']) / abs(pos['quantity'])
                current_price = st.session_state.price_history[symbol][-1]
                pos['current_price'] = current_price
                pos['unrealized_pnl'] = (current_price - pos['avg_price']) * pos['quantity']
                pos['market_value'] = current_price * pos['quantity']
        
        # Position Summary
        st.markdown("#### 📈 Position Summary")
        
        position_data = []
        total_pnl = 0
        total_market_value = 0
        
        for symbol, pos in positions.items():
            if pos['quantity'] != 0:
                total_pnl += pos['unrealized_pnl']
                total_market_value += pos['market_value']
                
                position_data.append({
                    'Symbol': symbol,
                    'Quantity': f"{pos['quantity']:,}",
                    'Avg Price': f"${pos['avg_price']:.2f}",
                    'Current Price': f"${pos['current_price']:.2f}",
                    'Market Value': f"${pos['market_value']:,.0f}",
                    'Unrealized P&L': f"${pos['unrealized_pnl']:,.0f}",
                    'Strategies': ', '.join(pos['strategies'])
                })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unrealized P&L", f"${total_pnl:,.0f}")
        with col2:
            st.metric("Total Market Value", f"${total_market_value:,.0f}")
        with col3:
            active_positions = len([p for p in positions.values() if p['quantity'] != 0])
            st.metric("Active Positions", active_positions)
        
        if position_data:
            st.dataframe(pd.DataFrame(position_data), use_container_width=True, hide_index=True)
        
        # Real-time Position Monitoring
        st.markdown("#### ⏱️ Real-time Position Monitoring")
        
        # Position risk metrics
        if positions:
            max_position_risk = max([abs(p['market_value']) for p in positions.values() if p['quantity'] != 0])
            risk_utilization = max_position_risk / (position_size * 5)  # Assuming 5x position limit
            
            col1, col2 = st.columns(2)
            with col1:
                if risk_utilization < 0.7:
                    st.success(f"✅ Position Risk: {risk_utilization*100:.1f}% utilized")
                elif risk_utilization < 0.9:
                    st.warning(f"⚠️ Position Risk: {risk_utilization*100:.1f}% utilized")
                else:
                    st.error(f"🔴 Position Risk: {risk_utilization*100:.1f}% utilized - LIMIT BREACH")
            
            with col2:
                # Position concentration
                total_gross = sum([abs(p['market_value']) for p in positions.values() if p['quantity'] != 0])
                if total_gross > 0:
                    max_concentration = max_position_risk / total_gross
                    st.metric("Max Position Concentration", f"{max_concentration*100:.1f}%")
    
    else:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>No Active Positions</strong><br>
            Execute some algorithmic trades to see position monitoring here.
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 📈 Post-Trade Analysis")
    
    if st.session_state.algo_trades:
        # Trade Performance Analysis
        st.markdown("#### 📊 Algorithm Performance Metrics")
        
        # Calculate performance by strategy
        strategy_performance = {}
        
        for trade in st.session_state.algo_trades:
            strategy = trade['strategy']
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    'trades': 0,
                    'total_notional': 0,
                    'avg_latency': 0,
                    'success_rate': 0
                }
            
            strategy_performance[strategy]['trades'] += 1
            strategy_performance[strategy]['total_notional'] += trade['notional']
            strategy_performance[strategy]['avg_latency'] += trade['latency_ms']
        
        # Finalize calculations
        for strategy, perf in strategy_performance.items():
            perf['avg_latency'] /= perf['trades']
            # Simulate success rate (in real system would track actual P&L)
            perf['success_rate'] = np.random.uniform(0.55, 0.75)
        
        # Performance table
        perf_data = []
        for strategy, perf in strategy_performance.items():
            perf_data.append({
                'Strategy': strategy,
                'Total Trades': perf['trades'],
                'Total Notional': f"${perf['total_notional']:,.0f}",
                'Avg Latency': f"{perf['avg_latency']:.1f}ms",
                'Success Rate': f"{perf['success_rate']*100:.1f}%",
                'Sharpe Ratio': f"{np.random.uniform(0.8, 1.5):.2f}"  # Simulated
            })
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
        
        # Execution Quality Analysis
        st.markdown("#### ⚡ Execution Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Latency distribution
            latencies = [t['latency_ms'] for t in st.session_state.algo_trades]
            
            fig = go.Figure(data=[go.Histogram(x=latencies, nbinsx=20)])
            fig.update_layout(
                title="Execution Latency Distribution",
                xaxis_title="Latency (ms)",
                yaxis_title="Count",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trade size distribution
            trade_sizes = [t['notional'] for t in st.session_state.algo_trades]
            
            fig = go.Figure(data=[go.Box(y=trade_sizes, name="Trade Size")])
            fig.update_layout(
                title="Trade Size Distribution",
                yaxis_title="Notional ($)",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Strategy Comparison
        st.markdown("#### 🔬 Strategy Comparison")
        
        if len(strategy_performance) > 1:
            strategies = list(strategy_performance.keys())
            metrics = ['trades', 'avg_latency', 'success_rate']
            
            fig = go.Figure()
            
            for i, metric in enumerate(metrics):
                values = [strategy_performance[s][metric] for s in strategies]
                if metric == 'success_rate':
                    values = [v * 100 for v in values]  # Convert to percentage
                
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=strategies,
                    y=values,
                    yaxis=f'y{i+1}' if i > 0 else 'y'
                ))
            
            fig.update_layout(
                title="Strategy Performance Comparison",
                template="plotly_white",
                height=400,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No trades executed yet. Post-trade analysis will appear after algorithmic trading.")

with tab4:
    st.markdown("### 🔄 Position Reconciliation")
    
    # Start of Day Reconciliation
    st.markdown("#### 🌅 Start of Day Reconciliation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Expected Positions (SOD):**")
        
        # Simulated SOD positions
        sod_positions = {
            'SPY': {'quantity': 1000, 'price': 449.50},
            'QQQ': {'quantity': -500, 'price': 375.25}
        }
        
        sod_data = []
        for symbol, pos in sod_positions.items():
            sod_data.append({
                'Symbol': symbol,
                'SOD Quantity': f"{pos['quantity']:,}",
                'SOD Price': f"${pos['price']:.2f}",
                'SOD Value': f"${pos['quantity'] * pos['price']:,.0f}"
            })
        
        st.dataframe(pd.DataFrame(sod_data), hide_index=True)
    
    with col2:
        st.markdown("**Current Positions:**")
        
        if st.session_state.algo_trades:
            # Calculate current positions (from algorithm tab logic)
            current_positions = {}
            for trade in st.session_state.algo_trades:
                symbol = trade['symbol']
                if symbol not in current_positions:
                    current_positions[symbol] = {'quantity': 0, 'avg_price': 0, 'total_value': 0}
                
                quantity = trade['quantity'] if trade['side'] == 'BUY' else -trade['quantity']
                current_positions[symbol]['quantity'] += quantity
                current_positions[symbol]['total_value'] += trade['notional'] if trade['side'] == 'BUY' else -trade['notional']
            
            current_data = []
            for symbol, pos in current_positions.items():
                if pos['quantity'] != 0:
                    avg_price = abs(pos['total_value']) / abs(pos['quantity'])
                    current_data.append({
                        'Symbol': symbol,
                        'Current Quantity': f"{pos['quantity']:,}",
                        'Avg Price': f"${avg_price:.2f}",
                        'Current Value': f"${pos['total_value']:,.0f}"
                    })
            
            if current_data:
                st.dataframe(pd.DataFrame(current_data), hide_index=True)
            else:
                st.info("No current positions")
        else:
            st.info("No current positions")
    
    # Reconciliation Status
    st.markdown("#### ✅ Reconciliation Status")
    
    reconciliation_items = [
        {"Item": "Position Quantities", "Status": "✅ Matched", "Variance": "0"},
        {"Item": "Cash Balances", "Status": "✅ Matched", "Variance": "$0"},
        {"Item": "Pending Settlements", "Status": "⚠️ Review", "Variance": "2 items"},
        {"Item": "Corporate Actions", "Status": "✅ Updated", "Variance": "0"},
        {"Item": "Exchange Confirmations", "Status": "✅ Received", "Variance": "0"}
    ]
    
    st.dataframe(pd.DataFrame(reconciliation_items), hide_index=True, use_container_width=True)
    
    # End of Day Reconciliation
    st.markdown("#### 🌆 End of Day Reconciliation")
    
    if st.button("🔍 Run EOD Reconciliation", type="primary"):
        with st.spinner("Running end-of-day reconciliation..."):
            time.sleep(2)
        
        st.success("✅ EOD Reconciliation Complete - All positions matched")
        
        # Reconciliation summary
        st.markdown("**Reconciliation Summary:**")
        st.markdown("• All algorithmic trades confirmed")
        st.markdown("• Position balances verified")
        st.markdown("• No breaks identified")
        st.markdown("• Ready for next trading session")

with tab5:
    st.markdown("### 📋 Profit & Loss Reporting")
    
    # Daily P&L Summary
    st.markdown("#### 💰 Daily P&L Summary")
    
    if st.session_state.algo_trades:
        # Calculate realized and unrealized P&L
        total_realized_pnl = 0
        total_unrealized_pnl = 0
        
        # Group trades by symbol to calculate unrealized P&L
        symbol_positions = {}
        
        for trade in st.session_state.algo_trades:
            symbol = trade['symbol']
            if symbol not in symbol_positions:
                symbol_positions[symbol] = {'quantity': 0, 'cost_basis': 0}
            
            quantity = trade['quantity'] if trade['side'] == 'BUY' else -trade['quantity']
            symbol_positions[symbol]['quantity'] += quantity
            symbol_positions[symbol]['cost_basis'] += trade['notional'] if trade['side'] == 'BUY' else -trade['notional']
        
        # Calculate unrealized P&L
        for symbol, pos in symbol_positions.items():
            if pos['quantity'] != 0:
                current_price = st.session_state.price_history[symbol][-1]
                avg_cost = abs(pos['cost_basis']) / abs(pos['quantity'])
                unrealized = (current_price - avg_cost) * pos['quantity']
                total_unrealized_pnl += unrealized
        
        # P&L Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Realized P&L", f"${total_realized_pnl:,.0f}")
        with col2:
            st.metric("Unrealized P&L", f"${total_unrealized_pnl:,.0f}")
        with col3:
            total_pnl = total_realized_pnl + total_unrealized_pnl
            st.metric("Total P&L", f"${total_pnl:,.0f}")
        with col4:
            total_notional = sum([t['notional'] for t in st.session_state.algo_trades])
            pnl_percentage = (total_pnl / total_notional * 100) if total_notional > 0 else 0
            st.metric("P&L %", f"{pnl_percentage:.2f}%")
        
        # P&L by Strategy
        st.markdown("#### 📊 P&L by Algorithm Strategy")
        
        strategy_pnl = {}
        for trade in st.session_state.algo_trades:
            strategy = trade['strategy']
            if strategy not in strategy_pnl:
                strategy_pnl[strategy] = {
                    'trades': 0,
                    'notional': 0,
                    'pnl': np.random.uniform(-5000, 15000)  # Simulated P&L
                }
            
            strategy_pnl[strategy]['trades'] += 1
            strategy_pnl[strategy]['notional'] += trade['notional']
        
        strategy_pnl_data = []
        for strategy, data in strategy_pnl.items():
            strategy_pnl_data.append({
                'Strategy': strategy,
                'Trades': data['trades'],
                'Notional': f"${data['notional']:,.0f}",
                'P&L': f"${data['pnl']:,.0f}",
                'Return %': f"{(data['pnl']/data['notional']*100):.2f}%"
            })
        
        st.dataframe(pd.DataFrame(strategy_pnl_data), hide_index=True, use_container_width=True)
        
        # P&L Chart
        st.markdown("#### 📈 P&L Trend")
        
        # Simulate P&L history
        pnl_timeline = []
        cumulative_pnl = 0
        
        for i, trade in enumerate(st.session_state.algo_trades):
            trade_pnl = np.random.uniform(-1000, 2000)  # Simulated trade P&L
            cumulative_pnl += trade_pnl
            pnl_timeline.append({
                'Trade': i + 1,
                'Timestamp': trade['timestamp'],
                'Trade P&L': trade_pnl,
                'Cumulative P&L': cumulative_pnl
            })
        
        if pnl_timeline:
            pnl_df = pd.DataFrame(pnl_timeline)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pnl_df['Trade'],
                y=pnl_df['Cumulative P&L'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#1a472a', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative P&L by Trade",
                xaxis_title="Trade Number",
                yaxis_title="P&L ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export P&L Report
        if st.button("📄 Generate P&L Report", type="secondary"):
            st.success("✅ P&L Report generated and saved to /reports/daily_pnl.pdf")
    
    else:
        st.info("No trading activity yet. P&L reporting will appear after algorithmic trading.")

with tab6:
    st.markdown("### 🔬 Strategy Research & Development")
    
    # Strategy Backtesting
    st.markdown("#### 📊 Strategy Backtesting")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Backtest Parameters:**")
        
        backtest_strategy = st.selectbox("Strategy to Test", 
            ["Mean Reversion", "Momentum", "Statistical Arbitrage"], key="backtest_strategy")
        
        backtest_period = st.selectbox("Backtest Period", 
            ["1 Week", "1 Month", "3 Months", "6 Months"])
        
        initial_capital = st.number_input("Initial Capital", value=1000000, min_value=100000, step=100000)
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest simulation..."):
                time.sleep(3)
            
            # Simulate backtest results
            backtest_results = {
                'Total Return': f"{np.random.uniform(5, 25):.2f}%",
                'Sharpe Ratio': f"{np.random.uniform(1.2, 2.5):.2f}",
                'Max Drawdown': f"{np.random.uniform(-5, -15):.2f}%",
                'Win Rate': f"{np.random.uniform(55, 75):.1f}%",
                'Total Trades': f"{np.random.randint(500, 2000):,}",
                'Avg Trade P&L': f"${np.random.uniform(50, 200):.0f}"
            }
            
            st.success("✅ Backtest Complete!")
            
            for metric, value in backtest_results.items():
                st.metric(metric, value)
    
    with col2:
        st.markdown("**Research Notes:**")
        
        research_notes = st.text_area(
            "Strategy Research Notes",
            value="• Testing mean reversion on SPY with 20-day lookback\n• Signal threshold: 1.5 standard deviations\n• Position sizing: $100k per trade\n• Risk management: 2% stop loss",
            height=200
        )
        
        if st.button("💾 Save Research Notes"):
            st.success("✅ Research notes saved to strategy database")
    
    # Strategy Performance Comparison
    st.markdown("#### 📈 Strategy Performance Comparison")
    
    # Simulate strategy comparison data
    strategies = ["Mean Reversion", "Momentum", "Statistical Arbitrage", "Market Making"]
    metrics = ["Return %", "Sharpe Ratio", "Max Drawdown %", "Win Rate %"]
    
    comparison_data = []
    for strategy in strategies:
        row = {'Strategy': strategy}
        row['Return %'] = f"{np.random.uniform(8, 20):.1f}%"
        row['Sharpe Ratio'] = f"{np.random.uniform(1.0, 2.2):.2f}"
        row['Max Drawdown %'] = f"{np.random.uniform(-8, -20):.1f}%"
        row['Win Rate %'] = f"{np.random.uniform(52, 72):.1f}%"
        comparison_data.append(row)
    
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
    
    # Research Tools
    st.markdown("#### 🛠️ Quantitative Research Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Correlation Analysis", use_container_width=True):
            st.info("Running correlation analysis on asset pairs...")
    
    with col2:
        if st.button("📈 Volatility Modeling", use_container_width=True):
            st.info("Fitting GARCH models to volatility data...")
    
    with col3:
        if st.button("🔍 Signal Discovery", use_container_width=True):
            st.info("Scanning for new alpha signals...")

# Real-time updates
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh All Data", use_container_width=True):
    # Update market data
    new_price = st.session_state.price_history['SPY'][-1] + np.random.normal(0, volatility * base_price / 100)
    st.session_state.price_history['SPY'].append(new_price)
    if len(st.session_state.price_history['SPY']) > 100:
        st.session_state.price_history['SPY'] = st.session_state.price_history['SPY'][-100:]
    
    st.success("All market data refreshed!")
    st.rerun()

# Enhanced Footer
st.markdown("""
<div class="footer">
    <h4>🦅 EAGLE SEVEN, LLC ALGORITHMIC TRADING SIMULATOR</h4>
    <p><strong>Built by Muchiri Kahwai</strong> | Demonstrating Algorithmic Trading & Operations</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        <em>Featuring: Algorithm Execution • Global Markets • Position Reconciliation • Post-Trade Analysis • PnL Reporting • Strategy Research</em>
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem; color: #6c757d;">
        📧 mk@Muchiri.tech | 📱 +1(859)319-6196 | 
        <a href="https://linkedin.com/in/muchiri-kahwai" style="color: #1a472a;">LinkedIn</a> | 
        <a href="https://github.com/muchirikahwai" style="color: #1a472a;">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)