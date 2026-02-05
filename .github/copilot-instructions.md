# 幸运数字预测系统 - AI Agent Instructions

## Project Overview

Chinese lottery prediction system using machine learning to predict lucky numbers (1-49) with associated Chinese zodiac animals (生肖) and five elements (五行: 金木水火土). The system achieves 50-63% hit rates across different predictor models.

## Architecture

### Core Prediction Models (30+ variants)
- **Base Models**: `lucky_number_predictor.py` - ML regression (RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost)
- **Zodiac Predictors**: Statistical models using Chinese zodiac patterns (`zodiac_*_predictor.py`)
  - Each zodiac has fixed number mappings: 鼠=[1,13,25,37,49], 牛=[2,14,26,38], etc.
- **Odd/Even Predictors**: Binary classification for odd/even numbers (`odd_even_predictor.py`)
- **Top15 Predictors**: Select 15 most likely numbers from 49 (`top15_*_predictor.py`, `ensemble_top15_predictor.py`)
- **Hybrid Models**: Combine multiple strategies (`hybrid_predictor.py`, `enhanced_hybrid_predictor.py`)
- **Betting Strategy**: Progressive betting system (Martingale, Fibonacci, D'Alembert) in `betting_strategy.py`

### Data Structure
**Primary data file**: `data/lucky_numbers.csv` with 4 required columns:
```csv
date,number,animal,element
2025/2/1,39,兔,火
```
- `number`: 1-49 lottery number
- `animal`: 12 Chinese zodiacs (鼠牛虎兔龙蛇马羊猴鸡狗猪)
- `element`: 5 elements (金木水火土 - Metal, Wood, Water, Fire, Earth)

### UI Layer
- **Main GUI**: `lucky_number_gui.py` (4576 lines) - Tkinter interface integrating all predictors
  - Auto-loads `data/lucky_numbers.csv` on startup
  - Supports multiple prediction modes: Top15, Zodiac, Odd/Even
  - Real-time validation and betting strategy analysis
  - Built-in model training, prediction, and visualization

## Key Development Patterns

### Time Series Windowing
All predictors use sliding window approach:
```python
def create_sequences(data, seq_length=10):
    # Convert [1,2,3,4,5] to features → target pairs
    # [1,2,3] → 4, [2,3,4] → 5
```

### Feature Engineering Standard
Extract from each window:
- Historical lag values (lag_1 to lag_n)
- Statistical features (mean, std, max, min)
- Trend features (recent changes)
- Zodiac/element distribution patterns
- Hot/cold number analysis (频率分析)

### Model Persistence
Standard pattern across all predictors:
```python
def save_model(self, model_dir='models'):
    # Save with timestamp: {ModelName}_{timestamp}.joblib
    model_data = {'model': self.model, 'scaler': self.scaler, ...}
    joblib.dump(model_data, filepath)

def load_model(self, model_file):
    model_data = joblib.load(model_file)
```
Models saved to `models/` directory with timestamp naming.

### Validation Pattern
All models follow rolling validation:
```python
def validate_predictor(predictor, numbers, test_periods=100):
    # Rolling window validation
    for i in range(len(numbers) - test_periods, len(numbers)):
        train_data = numbers[:i]
        actual = numbers[i]
        prediction = predictor.predict(train_data)
        # Calculate hit rates for Top5/Top15/Top20
```
Hit rate metrics: Top5, Top10, Top15, Top20 accuracy percentages.

## Critical Workflows

### Running Predictions (GUI)
```powershell
python lucky_number_gui.py
```
Auto-loads data, no manual data loading needed. Click prediction buttons for immediate results.

### Validation Testing (100 periods standard)
```powershell
python validate_*_100periods.py  # Specific model validation
python compare_models_10periods.py  # Quick comparison
```

### New Predictor Development
1. Inherit structure from `zodiac_super_predictor.py` or `top15_statistical_predictor.py`
2. Implement `predict(numbers)` returning predictions
3. Add to GUI imports and initialization
4. Create `validate_*.py` script for backtesting
5. Document results in markdown report (see `*验证报告.md` files)

### Betting Strategy Integration
```python
from betting_strategy import BettingStrategy
strategy = BettingStrategy(base_bet=15, win_reward=45)
multiplier, bet = strategy.calculate_martingale()  # Progressive betting
```
Strategies: Fixed, Martingale (aggressive), Fibonacci (balanced), D'Alembert (conservative)

## Project-Specific Conventions

### File Naming
- Predictors: `{type}_{variant}_predictor.py` (e.g., `zodiac_enhanced_60_predictor.py`)
- Validators: `validate_{model}_{periods}periods.py`
- Tests: `test_{feature}.py` or `quick_test_{feature}.py`
- Demos: `demo_{feature}.py` for user-facing examples
- Reports: Chinese markdown `{Feature}验证报告.md` or `{Feature}说明.md`

### Chinese-English Mix
- Code: English variables/functions
- Comments: Mix of English/Chinese
- Documentation: Primarily Chinese
- UI strings: Chinese with emoji decorators
- Model names: English with Chinese descriptions

### Version Progression
Model versions indicate iteration: v5.0, v10.0, v11.0, v12.0 (e.g., `zodiac_v10`, `zodiac_v11`)
Higher versions aren't always better - check validation reports for actual performance.

### Success Metrics
- **Top5 hit rate**: ≥50% considered successful
- **Top15 hit rate**: ≥60% target goal
- **Zodiac prediction**: 50-63% achieved across different models
- Always validate with 50-100 period rolling backtests

## Dependencies & Environment

Install: `pip install -r requirements.txt`
Key packages: numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, matplotlib, tkinter

Python 3.7+ required (use Windows PowerShell for commands)

## Important Files

**Entry points**:
- `lucky_number_gui.py` - Main application
- `main.py` - Alternative launcher

**Best performing models**:
- `zodiac_enhanced_60_predictor.py` - 63% success rate
- `ensemble_top15_predictor.py` - Ensemble approach
- `top15_statistical_predictor.py` - Statistical distribution analysis

**Core utilities**:
- `generate_data.py` - Generate sample data
- `betting_strategy.py` - Betting system calculations

## Testing Strategy

Before committing predictor changes:
1. Run 100-period validation: `python validate_{model}_100periods.py`
2. Compare with existing models: `python compare_models_10periods.py`
3. Test GUI integration: Click predict buttons in `lucky_number_gui.py`
4. Document results in Chinese markdown report
5. Update GUI title with new success rate if improved

## Common Pitfalls

- **Don't** mix up zodiac number mappings - they're fixed by Chinese astrology
- **Don't** validate on training data - always use rolling/forward validation
- **Don't** assume newer models perform better - validate first
- **Remember** data encoding: CSV files use UTF-8-sig for Chinese characters
- **Check** model directory exists before saving: `os.makedirs('models', exist_ok=True)`
- **GUI threading**: Use `threading.Thread` for predictions to keep UI responsive
