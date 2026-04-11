import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import hashlib
import os
from datetime import datetime
import calendar

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ExpenseAI - Smart Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
USER_FILE = "users.csv"
EXPENSE_FILE = "expenses.csv"
BUDGET_FILE = "budgets.csv"

# Professional category definitions
CATEGORIES = {
    "Housing": {
        "icon": "🏠",
        "keywords": ["rent", "mortgage", "property", "maintenance", "repair", "plumber", "electrician", "furniture", "appliance", "home"],
        "color": "#FF6B6B"
    },
    "Transportation": {
        "icon": "🚗",
        "keywords": ["uber", "ola", "lyft", "taxi", "bus", "train", "metro", "flight", "petrol", "diesel", "fuel", "gas", "parking", "toll", "car", "bike", "vehicle", "insurance"],
        "color": "#4ECDC4"
    },
    "Food & Dining": {
        "icon": "🍽️",
        "keywords": ["restaurant", "food", "grocery", "groceries", "supermarket", "vegetables", "fruits", "meat", "fish", "swiggy", "zomato", "doordash", "ubereats", "cafe", "coffee", "tea", "breakfast", "lunch", "dinner", "snacks", "pizza", "burger"],
        "color": "#45B7D1"
    },
    "Utilities": {
        "icon": "💡",
        "keywords": ["electricity", "water", "gas", "internet", "wifi", "broadband", "phone", "mobile", "recharge", "bill", "utility"],
        "color": "#96CEB4"
    },
    "Healthcare": {
        "icon": "🏥",
        "keywords": ["doctor", "hospital", "medicine", "pharmacy", "medical", "health", "dental", "dentist", "eye", "glasses", "therapy", "consultation", "lab", "test", "insurance", "gym", "fitness"],
        "color": "#FFEAA7"
    },
    "Shopping": {
        "icon": "🛍️",
        "keywords": ["amazon", "flipkart", "myntra", "clothes", "clothing", "shoes", "fashion", "accessories", "electronics", "gadget", "laptop", "phone", "watch", "jewelry", "cosmetics", "beauty"],
        "color": "#DDA0DD"
    },
    "Entertainment": {
        "icon": "🎬",
        "keywords": ["movie", "cinema", "netflix", "spotify", "prime", "disney", "subscription", "concert", "event", "game", "gaming", "sports", "hobby", "book", "magazine"],
        "color": "#F39C12"
    },
    "Education": {
        "icon": "📚",
        "keywords": ["course", "class", "tuition", "school", "college", "university", "book", "stationery", "exam", "certification", "training", "workshop", "seminar", "udemy", "coursera"],
        "color": "#9B59B6"
    },
    "Personal Care": {
        "icon": "💅",
        "keywords": ["salon", "spa", "haircut", "grooming", "skincare", "makeup", "perfume", "hygiene"],
        "color": "#E91E63"
    },
    "Insurance": {
        "icon": "🛡️",
        "keywords": ["life insurance", "health insurance", "car insurance", "home insurance", "premium", "policy"],
        "color": "#607D8B"
    },
    "Investments": {
        "icon": "📈",
        "keywords": ["mutual fund", "stocks", "shares", "sip", "investment", "deposit", "fd", "rd", "bonds", "crypto", "bitcoin"],
        "color": "#2ECC71"
    },
    "Gifts & Donations": {
        "icon": "🎁",
        "keywords": ["gift", "present", "donation", "charity", "wedding", "birthday", "anniversary", "festival"],
        "color": "#E74C3C"
    },
    "Travel": {
        "icon": "✈️",
        "keywords": ["vacation", "holiday", "trip", "hotel", "airbnb", "booking", "resort", "tourism", "luggage", "visa", "passport"],
        "color": "#1ABC9C"
    },
    "EMI & Loans": {
        "icon": "💳",
        "keywords": ["emi", "loan", "credit card", "interest", "payment", "installment", "debt"],
        "color": "#34495E"
    },
    "Miscellaneous": {
        "icon": "📦",
        "keywords": ["other", "misc", "miscellaneous"],
        "color": "#95A5A6"
    }
}

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Transaction table styling */
    .transaction-row {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Category badge */
    .category-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #e0e0e0, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def hash_password(password: str) -> str:
    """Securely hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_files():
    """Initialize CSV files if they don't exist."""
    if not os.path.exists(USER_FILE):
        pd.DataFrame(columns=["username", "password_hash", "created_at"]).to_csv(USER_FILE, index=False)
    
    if not os.path.exists(EXPENSE_FILE):
        pd.DataFrame(columns=[
            "user", "description", "category", "amount", 
            "date", "month", "year", "payment_method", "notes"
        ]).to_csv(EXPENSE_FILE, index=False)
    
    if not os.path.exists(BUDGET_FILE):
        pd.DataFrame(columns=["user", "category", "monthly_limit"]).to_csv(BUDGET_FILE, index=False)

def format_currency(amount: float, currency: str = "₹") -> str:
    """Format amount as currency with proper formatting."""
    if amount >= 10000000:  # 1 Crore
        return f"{currency}{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"{currency}{amount/100000:.2f} L"
    elif amount >= 1000:
        return f"{currency}{amount/1000:.1f}K"
    return f"{currency}{amount:,.0f}"

def get_month_name(month_num: int) -> str:
    """Convert month number to name."""
    return calendar.month_name[month_num]

def get_greeting() -> str:
    """Return time-appropriate greeting."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning"
    elif hour < 17:
        return "Good Afternoon"
    return "Good Evening"

# ============================================================
# AI CATEGORY PREDICTOR
# ============================================================

class CategoryPredictor:
    """AI-powered expense category prediction."""
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self._train_model()
    
    def _train_model(self):
        """Train the model with comprehensive training data."""
        training_data = []
        labels = []
        
        for category, info in CATEGORIES.items():
            # Create multiple training samples per category
            keywords = info["keywords"]
            for keyword in keywords:
                training_data.append(keyword)
                labels.append(category)
            # Add combined phrases
            training_data.append(" ".join(keywords))
            labels.append(category)
        
        self.pipeline.fit(training_data, labels)
    
    def predict(self, description: str) -> tuple:
        """Predict category with confidence score."""
        if not description.strip():
            return "Miscellaneous", 0.0
        
        try:
            prediction = self.pipeline.predict([description.lower()])[0]
            probabilities = self.pipeline.predict_proba([description.lower()])[0]
            confidence = max(probabilities) * 100
            return prediction, confidence
        except Exception:
            return "Miscellaneous", 0.0

# ============================================================
# AUTHENTICATION
# ============================================================

def render_auth_page():
    """Render login/signup page with professional styling."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">💰 ExpenseAI</h1>', unsafe_allow_html=True)
        st.markdown("#### Smart AI-Powered Expense Tracking")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        users = pd.read_csv(USER_FILE)
                        password_hash = hash_password(password)
                        
                        if ((users["username"] == username) & (users["password_hash"] == password_hash)).any():
                            st.session_state.user = username
                            st.session_state.logged_in = True
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please fill in all fields")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="Enter a username")
                new_password = st.text_input("Choose Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                submit = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit:
                    if not all([new_username, new_password, confirm_password]):
                        st.warning("Please fill in all fields")
                    elif len(new_password) < 6:
                        st.warning("Password must be at least 6 characters")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        users = pd.read_csv(USER_FILE)
                        if new_username in users["username"].values:
                            st.error("Username already exists")
                        else:
                            new_user = pd.DataFrame({
                                "username": [new_username],
                                "password_hash": [hash_password(new_password)],
                                "created_at": [datetime.now().isoformat()]
                            })
                            users = pd.concat([users, new_user], ignore_index=True)
                            users.to_csv(USER_FILE, index=False)
                            st.success("Account created! Please login.")

# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard(user_data: pd.DataFrame):
    """Render the main dashboard with analytics."""
    
    st.markdown(f'<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f"### {get_greeting()}, {st.session_state.user}! 👋")
    
    # Current month stats
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    if not user_data.empty:
        user_data['date'] = pd.to_datetime(user_data['date'], errors='coerce')
        monthly_data = user_data[
            (user_data['date'].dt.month == current_month) & 
            (user_data['date'].dt.year == current_year)
        ]
    else:
        monthly_data = pd.DataFrame()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    total_all_time = user_data['amount'].sum() if not user_data.empty else 0
    total_this_month = monthly_data['amount'].sum() if not monthly_data.empty else 0
    transaction_count = len(monthly_data) if not monthly_data.empty else 0
    avg_transaction = monthly_data['amount'].mean() if not monthly_data.empty else 0
    
    with col1:
        st.metric(
            label="This Month",
            value=format_currency(total_this_month),
            delta=f"{transaction_count} transactions"
        )
    
    with col2:
        st.metric(
            label="All Time Spending",
            value=format_currency(total_all_time)
        )
    
    with col3:
        st.metric(
            label="Avg. Transaction",
            value=format_currency(avg_transaction)
        )
    
    with col4:
        # Calculate month-over-month change
        if not user_data.empty:
            last_month_data = user_data[
                (user_data['date'].dt.month == (current_month - 1 if current_month > 1 else 12)) & 
                (user_data['date'].dt.year == (current_year if current_month > 1 else current_year - 1))
            ]
            last_month_total = last_month_data['amount'].sum()
            if last_month_total > 0:
                change = ((total_this_month - last_month_total) / last_month_total) * 100
                st.metric(
                    label="vs Last Month",
                    value=f"{change:+.1f}%",
                    delta="spending change"
                )
            else:
                st.metric(label="vs Last Month", value="N/A")
        else:
            st.metric(label="vs Last Month", value="N/A")
    
    st.markdown("---")
    
    if not user_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💸 Spending by Category")
            category_data = user_data.groupby('category')['amount'].sum().reset_index()
            category_data = category_data.sort_values('amount', ascending=False)
            
            # Add icons and colors
            category_data['icon'] = category_data['category'].map(
                lambda x: CATEGORIES.get(x, {}).get('icon', '📦')
            )
            category_data['color'] = category_data['category'].map(
                lambda x: CATEGORIES.get(x, {}).get('color', '#95A5A6')
            )
            
            fig = px.pie(
                category_data, 
                values='amount', 
                names='category',
                color='category',
                color_discrete_map={cat: info['color'] for cat, info in CATEGORIES.items()},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Monthly Trend")
            monthly_trend = user_data.groupby(
                user_data['date'].dt.to_period('M')
            )['amount'].sum().reset_index()
            monthly_trend['date'] = monthly_trend['date'].astype(str)
            
            fig = px.bar(
                monthly_trend,
                x='date',
                y='amount',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                showlegend=False,
                margin=dict(t=0, b=0),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent transactions
        st.subheader("🕐 Recent Transactions")
        recent = user_data.sort_values('date', ascending=False).head(5)
        
        for _, row in recent.iterrows():
            icon = CATEGORIES.get(row['category'], {}).get('icon', '📦')
            color = CATEGORIES.get(row['category'], {}).get('color', '#95A5A6')
            
            col1, col2, col3, col4 = st.columns([0.5, 3, 2, 1.5])
            with col1:
                st.write(icon)
            with col2:
                st.write(f"**{row['description'][:30]}**")
            with col3:
                st.write(f"`{row['category']}`")
            with col4:
                st.write(f"**{format_currency(row['amount'])}**")
    else:
        st.info("No expenses recorded yet. Start by adding your first expense! ➡️")

# ============================================================
# ADD EXPENSE
# ============================================================

def render_add_expense(predictor: CategoryPredictor):
    """Render the add expense form."""
    
    st.markdown('<h1 class="main-header">➕ Add Expense</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("expense_form"):
            description = st.text_input(
                "Description *",
                placeholder="e.g., Grocery shopping at BigBasket",
                help="Describe your expense - our AI will auto-categorize it"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                amount = st.number_input(
                    "Amount (₹) *",
                    min_value=0.0,
                    step=10.0,
                    format="%.2f"
                )
            
            with col_b:
                date = st.date_input(
                    "Date *",
                    value=datetime.now()
                )
            
            # AI Category Prediction
            if description:
                predicted_category, confidence = predictor.predict(description)
                st.info(f"🤖 AI Suggestion: **{predicted_category}** ({confidence:.0f}% confidence)")
            else:
                predicted_category = "Miscellaneous"
            
            category = st.selectbox(
                "Category",
                options=list(CATEGORIES.keys()),
                index=list(CATEGORIES.keys()).index(predicted_category) if predicted_category in CATEGORIES else 0,
                format_func=lambda x: f"{CATEGORIES[x]['icon']} {x}"
            )
            
            payment_method = st.selectbox(
                "Payment Method",
                options=["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking", "Wallet"]
            )
            
            notes = st.text_area(
                "Notes (Optional)",
                placeholder="Any additional details...",
                height=80
            )
            
            submitted = st.form_submit_button("💾 Save Expense", use_container_width=True)
            
            if submitted:
                if not description or amount <= 0:
                    st.error("Please fill in description and amount")
                else:
                    expenses = pd.read_csv(EXPENSE_FILE)
                    new_expense = pd.DataFrame({
                        "user": [st.session_state.user],
                        "description": [description],
                        "category": [category],
                        "amount": [amount],
                        "date": [date.isoformat()],
                        "month": [date.month],
                        "year": [date.year],
                        "payment_method": [payment_method],
                        "notes": [notes]
                    })
                    expenses = pd.concat([expenses, new_expense], ignore_index=True)
                    expenses.to_csv(EXPENSE_FILE, index=False)
                    st.success(f"✅ Expense saved: {format_currency(amount)} for {description}")
                    st.balloons()
    
    with col2:
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Be descriptive for better AI predictions
        - Review the suggested category
        - Add notes for recurring expenses
        - Track all payment methods
        """)
        
        st.markdown("### 📊 Category Guide")
        for cat, info in list(CATEGORIES.items())[:6]:
            st.markdown(f"{info['icon']} **{cat}**")

# ============================================================
# INSIGHTS
# ============================================================

def render_insights(user_data: pd.DataFrame):
    """Render AI-powered insights and predictions."""
    
    st.markdown('<h1 class="main-header">📈 AI Insights</h1>', unsafe_allow_html=True)
    
    if user_data.empty:
        st.info("Add some expenses to see AI-powered insights!")
        return
    
    user_data['date'] = pd.to_datetime(user_data['date'], errors='coerce')
    
    # Spending Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Spending Analysis")
        category_analysis = user_data.groupby('category')['amount'].agg(['sum', 'mean', 'count']).reset_index()
        category_analysis.columns = ['Category', 'Total', 'Average', 'Count']
        category_analysis = category_analysis.sort_values('Total', ascending=False)
        
        fig = px.bar(
            category_analysis,
            x='Total',
            y='Category',
            orientation='h',
            color='Category',
            color_discrete_map={cat: info['color'] for cat, info in CATEGORIES.items()}
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Spending Patterns")
        
        # Day of week analysis
        user_data['day_of_week'] = user_data['date'].dt.day_name()
        dow_spending = user_data.groupby('day_of_week')['amount'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig = px.line(
            x=dow_spending.index,
            y=dow_spending.values,
            markers=True,
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Total Spending (₹)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # AI Recommendations
    st.subheader("💡 AI Recommendations")
    
    top_category = category_analysis.iloc[0]['Category']
    top_spending = category_analysis.iloc[0]['Total']
    total_spending = category_analysis['Total'].sum()
    top_percentage = (top_spending / total_spending) * 100
    
    recommendations = {
        "Housing": [
            "Consider refinancing if interest rates have dropped",
            "Look for ways to reduce utility costs",
            "Regular maintenance prevents costly repairs"
        ],
        "Food & Dining": [
            "Meal planning can reduce food waste and costs",
            "Cooking at home is typically 3-5x cheaper than dining out",
            "Buy groceries in bulk for frequently used items"
        ],
        "Transportation": [
            "Consider carpooling or public transport",
            "Regular vehicle maintenance improves fuel efficiency",
            "Compare fuel prices using apps"
        ],
        "Shopping": [
            "Wait 24-48 hours before making impulse purchases",
            "Use price comparison tools before buying",
            "Set a monthly shopping budget"
        ],
        "Entertainment": [
            "Review your subscriptions - cancel unused ones",
            "Look for free local events and activities",
            "Share streaming subscriptions with family"
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white;">
            <h3>Top Spending Category</h3>
            <h2>{CATEGORIES.get(top_category, {}).get('icon', '📦')} {top_category}</h2>
            <p>{format_currency(top_spending)} ({top_percentage:.1f}% of total)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Personalized Tips")
        tips = recommendations.get(top_category, [
            "Set spending limits for each category",
            "Review expenses weekly",
            "Track all small purchases"
        ])
        for tip in tips:
            st.markdown(f"✅ {tip}")
    
    # Prediction
    st.markdown("---")
    st.subheader("🔮 Spending Forecast")
    
    if len(user_data) >= 10:
        # Prepare data for prediction
        monthly_totals = user_data.groupby(
            user_data['date'].dt.to_period('M')
        )['amount'].sum().reset_index()
        monthly_totals['month_num'] = range(len(monthly_totals))
        
        if len(monthly_totals) >= 3:
            # Train model
            X = monthly_totals[['month_num']].values
            y = monthly_totals['amount'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict next 3 months
            future_months = np.array([[len(monthly_totals) + i] for i in range(3)])
            predictions = model.predict(future_months)
            
            col1, col2, col3 = st.columns(3)
            months_ahead = ['Next Month', 'In 2 Months', 'In 3 Months']
            
            for i, (col, month_label) in enumerate(zip([col1, col2, col3], months_ahead)):
                with col:
                    st.metric(
                        label=month_label,
                        value=format_currency(max(0, predictions[i]))
                    )
            
            st.caption("*Predictions based on your historical spending patterns")
        else:
            st.info("Need at least 3 months of data for accurate predictions")
    else:
        st.info("Add more expenses to unlock spending predictions")

# ============================================================
# CHATBOT
# ============================================================

def render_chatbot(user_data: pd.DataFrame):
    """Render AI financial assistant chatbot."""
    
    st.markdown('<h1 class="main-header">💬 AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your expenses..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        response = generate_chatbot_response(prompt, user_data)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 💡 Try asking:")
    
    suggestions = [
        "What's my total spending this month?",
        "Which category do I spend most on?",
        "How can I save money?",
        "Show my spending summary",
        "Give me budgeting tips"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions[:3]):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

def generate_chatbot_response(query: str, user_data: pd.DataFrame) -> str:
    """Generate intelligent response based on user query and data."""
    
    query = query.lower()
    
    if user_data.empty:
        return "I don't see any expenses recorded yet. Start by adding your first expense to get personalized insights! 📝"
    
    total = user_data['amount'].sum()
    
    # Current month analysis
    user_data['date'] = pd.to_datetime(user_data['date'], errors='coerce')
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    monthly_data = user_data[
        (user_data['date'].dt.month == current_month) & 
        (user_data['date'].dt.year == current_year)
    ]
    monthly_total = monthly_data['amount'].sum()
    
    # Category analysis
    category_totals = user_data.groupby('category')['amount'].sum()
    top_category = category_totals.idxmax()
    top_amount = category_totals.max()
    
    if any(word in query for word in ['total', 'spent', 'spending', 'how much']):
        if 'month' in query or 'this month' in query:
            return f"""📊 **This Month's Summary**

You've spent **{format_currency(monthly_total)}** this month across {len(monthly_data)} transactions.

Your average transaction is {format_currency(monthly_data['amount'].mean() if len(monthly_data) > 0 else 0)}.

Would you like tips on how to reduce spending?"""
        else:
            return f"""💰 **Total Spending Overview**

Your all-time total spending is **{format_currency(total)}**.

This month: {format_currency(monthly_total)}
Total transactions: {len(user_data)}

Top spending category: {CATEGORIES.get(top_category, {}).get('icon', '📦')} {top_category}"""
    
    elif any(word in query for word in ['most', 'highest', 'top category', 'where']):
        return f"""🎯 **Spending Analysis**

Your highest spending category is **{CATEGORIES.get(top_category, {}).get('icon', '📦')} {top_category}** at {format_currency(top_amount)}.

This represents {(top_amount/total)*100:.1f}% of your total spending.

**Category Breakdown:**
{chr(10).join([f"• {CATEGORIES.get(cat, {}).get('icon', '📦')} {cat}: {format_currency(amt)}" for cat, amt in category_totals.sort_values(ascending=False).head(5).items()])}"""
    
    elif any(word in query for word in ['save', 'saving', 'reduce', 'cut']):
        return f"""💡 **Personalized Saving Tips**

Based on your spending in **{top_category}**, here are some suggestions:

1. **Set a Budget**: Allocate a fixed amount for {top_category} each month
2. **Track Daily**: Small daily savings compound over time
3. **50/30/20 Rule**: 50% needs, 30% wants, 20% savings
4. **Review Subscriptions**: Cancel unused services
5. **Wait Before Buying**: Apply 24-48 hour rule for non-essentials

Potential monthly savings by reducing {top_category} by 20%: **{format_currency(top_amount * 0.2)}**"""
    
    elif any(word in query for word in ['summary', 'overview', 'report']):
        return f"""📈 **Expense Summary Report**

**Overall Statistics:**
• Total Spent: {format_currency(total)}
• This Month: {format_currency(monthly_total)}
• Transactions: {len(user_data)}
• Avg. Transaction: {format_currency(user_data['amount'].mean())}

**Top Categories:**
{chr(10).join([f"• {CATEGORIES.get(cat, {}).get('icon', '📦')} {cat}: {format_currency(amt)}" for cat, amt in category_totals.sort_values(ascending=False).head(3).items()])}

**Insight:** Your {top_category} expenses are {(top_amount/total)*100:.0f}% of total spending."""
    
    elif any(word in query for word in ['budget', 'budgeting', 'plan']):
        return """📋 **Budgeting Tips**

**The 50/30/20 Rule:**
• 50% for Needs (Housing, Utilities, Groceries)
• 30% for Wants (Entertainment, Dining out)
• 20% for Savings & Debt repayment

**Smart Budgeting Steps:**
1. Track every expense for a month
2. Categorize spending patterns
3. Identify areas to cut back
4. Set realistic category limits
5. Review and adjust monthly

Would you like me to help create a personalized budget based on your spending?"""
    
    elif any(word in query for word in ['hello', 'hi', 'hey']):
        return f"""👋 Hello! I'm your AI Financial Assistant.

Here's what I can help you with:
• 📊 Spending summaries and analysis
• 💰 Tracking total expenses
• 🎯 Category breakdowns
• 💡 Personalized saving tips
• 📋 Budgeting advice

What would you like to know about your finances?"""
    
    else:
        return """🤔 I can help you with:

• **"What's my total spending?"** - See overall expenses
• **"Which category do I spend most on?"** - Category analysis
• **"How can I save money?"** - Personalized tips
• **"Show my summary"** - Complete overview
• **"Budgeting tips"** - Financial advice

Try asking one of these questions!"""

# ============================================================
# EXPENSE HISTORY
# ============================================================

def render_history(user_data: pd.DataFrame):
    """Render expense history with filters and management."""
    
    st.markdown('<h1 class="main-header">📜 Expense History</h1>', unsafe_allow_html=True)
    
    if user_data.empty:
        st.info("No expenses recorded yet. Start tracking your expenses!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.multiselect(
            "Filter by Category",
            options=list(CATEGORIES.keys()),
            default=[]
        )
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=[],
            key="date_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            options=["Date (Newest)", "Date (Oldest)", "Amount (High)", "Amount (Low)"]
        )
    
    # Apply filters
    filtered_data = user_data.copy()
    filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')
    
    if category_filter:
        filtered_data = filtered_data[filtered_data['category'].isin(category_filter)]
    
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data['date'].dt.date >= date_range[0]) &
            (filtered_data['date'].dt.date <= date_range[1])
        ]
    
    # Apply sorting
    if sort_by == "Date (Newest)":
        filtered_data = filtered_data.sort_values('date', ascending=False)
    elif sort_by == "Date (Oldest)":
        filtered_data = filtered_data.sort_values('date', ascending=True)
    elif sort_by == "Amount (High)":
        filtered_data = filtered_data.sort_values('amount', ascending=False)
    else:
        filtered_data = filtered_data.sort_values('amount', ascending=True)
    
    # Summary
    st.markdown(f"**Showing {len(filtered_data)} expenses | Total: {format_currency(filtered_data['amount'].sum())}**")
    
    st.markdown("---")
    
    # Display expenses
    for idx, row in filtered_data.iterrows():
        icon = CATEGORIES.get(row['category'], {}).get('icon', '📦')
        color = CATEGORIES.get(row['category'], {}).get('color', '#95A5A6')
        
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([0.5, 3, 2, 1.5, 0.5])
            
            with col1:
                st.write(icon)
            with col2:
                st.write(f"**{row['description']}**")
                st.caption(f"{row.get('payment_method', 'N/A')} • {row['date'].strftime('%d %b %Y') if pd.notna(row['date']) else 'N/A'}")
            with col3:
                st.markdown(f"`{row['category']}`")
            with col4:
                st.write(f"**{format_currency(row['amount'])}**")
            with col5:
                if st.button("🗑️", key=f"del_{idx}"):
                    expenses = pd.read_csv(EXPENSE_FILE)
                    expenses = expenses.drop(expenses[
                        (expenses['user'] == st.session_state.user) &
                        (expenses['description'] == row['description']) &
                        (expenses['amount'] == row['amount'])
                    ].index[:1])
                    expenses.to_csv(EXPENSE_FILE, index=False)
                    st.rerun()
            
            st.markdown("---")

# ============================================================
# BUDGET SETTINGS
# ============================================================

def render_budget_settings():
    """Render budget settings page."""
    
    st.markdown('<h1 class="main-header">⚙️ Budget Settings</h1>', unsafe_allow_html=True)
    
    budgets = pd.read_csv(BUDGET_FILE)
    user_budgets = budgets[budgets['user'] == st.session_state.user]
    
    st.subheader("Set Monthly Budgets")
    
    with st.form("budget_form"):
        budget_inputs = {}
        
        cols = st.columns(3)
        for i, (category, info) in enumerate(CATEGORIES.items()):
            with cols[i % 3]:
                existing = user_budgets[user_budgets['category'] == category]['monthly_limit'].values
                default_val = float(existing[0]) if len(existing) > 0 else 0.0
                
                budget_inputs[category] = st.number_input(
                    f"{info['icon']} {category}",
                    min_value=0.0,
                    value=default_val,
                    step=500.0,
                    key=f"budget_{category}"
                )
        
        if st.form_submit_button("💾 Save Budgets", use_container_width=True):
            # Remove existing budgets for user
            budgets = budgets[budgets['user'] != st.session_state.user]
            
            # Add new budgets
            for category, limit in budget_inputs.items():
                if limit > 0:
                    new_budget = pd.DataFrame({
                        'user': [st.session_state.user],
                        'category': [category],
                        'monthly_limit': [limit]
                    })
                    budgets = pd.concat([budgets, new_budget], ignore_index=True)
            
            budgets.to_csv(BUDGET_FILE, index=False)
            st.success("✅ Budgets saved successfully!")

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render the sidebar navigation."""
    
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <h2>💰 ExpenseAI</h2>
            <p style="color: gray;">Welcome, {st.session_state.user}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        menu = st.radio(
            "Navigation",
            options=[
                "📊 Dashboard",
                "➕ Add Expense",
                "📜 History",
                "📈 Insights",
                "💬 AI Assistant",
                "⚙️ Budget Settings"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")
        
        return menu

# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application entry point."""
    
    # Initialize files
    initialize_files()
    
    # Check authentication
    if "user" not in st.session_state or not st.session_state.get("logged_in"):
        render_auth_page()
        st.stop()
    
    # Initialize AI predictor
    predictor = CategoryPredictor()
    
    # Load user data
    all_expenses = pd.read_csv(EXPENSE_FILE)
    user_data = all_expenses[all_expenses['user'] == st.session_state.user].copy()
    
    # Render sidebar and get selected menu
    menu = render_sidebar()
    
    # Route to appropriate page
    if menu == "📊 Dashboard":
        render_dashboard(user_data)
    elif menu == "➕ Add Expense":
        render_add_expense(predictor)
    elif menu == "📜 History":
        render_history(user_data)
    elif menu == "📈 Insights":
        render_insights(user_data)
    elif menu == "💬 AI Assistant":
        render_chatbot(user_data)
    elif menu == "⚙️ Budget Settings":
        render_budget_settings()

if __name__ == "__main__":
    main()
