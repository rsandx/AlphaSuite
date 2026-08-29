from datetime import datetime
import sqlalchemy as sa
from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, BigInteger, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import class_mapper, relationship
from sqlalchemy import ForeignKey
from sqlalchemy.event import listens_for
from sqlalchemy.engine import Engine
from sqlalchemy.schema import Sequence  # Import Sequence

Base = declarative_base()

def object_as_dict(obj):
    """
    Converts a SQLAlchemy object to a dictionary.
    Handles relationships and nested objects.
    """
    if isinstance(obj, dict):
        return obj
    
    if hasattr(obj, '__table__'):
        # Check if obj is a class or an instance
        if isinstance(obj, type):
            return {}  # Return empty dict for class
        else:
            # Use instance_state to get the class mapper
            mapper = class_mapper(obj.__class__)
            return {c.key: getattr(obj, c.key) for c in mapper.columns}
    else:
        return obj


class Company(Base):
    __tablename__ = "company"

    id = Column(Integer, Sequence('company_id_seq', start=1, increment=1), primary_key=True)
    symbol = Column(String(255), unique=True, index=True)
    isactive = Column(Boolean, nullable=False, server_default=sa.sql.expression.true())
    address1 = Column(String(255))
    address2 = Column(String(255))
    city = Column(String(255))
    state = Column(String(255))
    zip = Column(String(255))
    country = Column(String(255))
    phone = Column(String(255))
    website = Column(String(255))
    industry = Column(String(255))
    industrykey = Column(String(255))
    industrydisp = Column(String(255))
    sector = Column(String(255))
    sectorkey = Column(String(255))
    sectordisp = Column(String(255))
    longbusinesssummary = Column(Text)
    fulltimeemployees = Column(Integer)
    auditrisk = Column(Integer)
    boardrisk = Column(Integer)
    compensationrisk = Column(Integer)
    shareholderrightsrisk = Column(Integer)
    overallrisk = Column(Integer)
    governanceepochdate = Column(Date) 
    compensationasofepochdate = Column(Date) 
    maxage = Column(Integer)
    pricehint = Column(Integer)
    previousclose = Column(Float)
    open = Column(Float)
    daylow = Column(Float)
    dayhigh = Column(Float)
    regularmarketpreviousclose = Column(Float)
    regularmarketopen = Column(Float)
    regularmarketdaylow = Column(Float)
    regularmarketdayhigh = Column(Float)
    dividendrate = Column(Float)
    dividendyield = Column(Float)
    exdividenddate = Column(Date) 
    payoutratio = Column(Float)
    beta = Column(Float)
    trailingpe = Column(Float)
    forwardpe = Column(Float, nullable=True)
    volume = Column(BigInteger)
    regularmarketvolume = Column(BigInteger)
    averagevolume = Column(BigInteger)
    averagevolume10days = Column(BigInteger)
    averagedailyvolume10day = Column(BigInteger)
    bid = Column(Float)
    ask = Column(Float)
    bidsize = Column(Integer)
    asksize = Column(Integer)
    marketcap = Column(BigInteger)
    fiftytwoweeklow = Column(Float)
    fiftytwoweekhigh = Column(Float)
    pricetosalestrailing12months = Column(Float)
    fiftydayaverage = Column(Float)
    twohundreddayaverage = Column(Float)
    trailingannualdividendrate = Column(Float)
    trailingannualdividendyield = Column(Float)
    currency = Column(String(10))
    tradeable = Column(Boolean)
    enterprisevalue = Column(BigInteger)
    profitmargins = Column(Float)
    floatshares = Column(BigInteger)
    sharesoutstanding = Column(BigInteger)
    sharesshort = Column(BigInteger)
    sharesshortpriormonth = Column(BigInteger)
    sharesshortpreviousmonthdate = Column(Date) 
    dateshortinterest = Column(Date) 
    sharespercentsharesout = Column(Float)
    heldpercentinsiders = Column(Float)
    heldpercentinstitutions = Column(Float)
    shortratio = Column(Float)
    shortpercentoffloat = Column(Float)
    impliedsharesoutstanding = Column(BigInteger)
    bookvalue = Column(Float)
    pricetobook = Column(Float)
    lastfiscalyearend = Column(Date) 
    nextfiscalyearend = Column(Date) 
    mostrecentquarter = Column(Date) 
    earningsquarterlygrowth = Column(Float)
    netincometocommon = Column(BigInteger)
    trailingeps = Column(Float)
    forwardeps = Column(Float)
    enterprisetorevenue = Column(Float)
    enterprisetoebitda = Column(Float)
    _52weekchange = Column(Float, nullable=True)
    sandp52weekchange = Column(Float)
    lastdividendvalue = Column(Float)
    lastdividenddate = Column(Date) 
    quotetype = Column(String(255))
    currentprice = Column(Float)
    targethighprice = Column(Float)
    targetlowprice = Column(Float)
    targetmeanprice = Column(Float)
    targetmedianprice = Column(Float)
    recommendationmean = Column(Float)
    recommendationkey = Column(String(255))
    numberofanalystopinions = Column(Integer)
    totalcash = Column(BigInteger)
    totalcashpershare = Column(Float)
    ebitda = Column(BigInteger)
    totaldebt = Column(BigInteger)
    quickratio = Column(Float)
    currentratio = Column(Float)
    totalrevenue = Column(BigInteger)
    debttoequity = Column(Float)
    revenuepershare = Column(Float)
    returnonassets = Column(Float)
    returnonequity = Column(Float)
    grossprofits = Column(BigInteger)
    freecashflow = Column(BigInteger)
    operatingcashflow = Column(BigInteger)
    earningsgrowth = Column(Float)
    revenuegrowth = Column(Float)
    grossmargins = Column(Float)
    ebitdamargins = Column(Float)
    operatingmargins = Column(Float)
    financialcurrency = Column(String(10))
    language = Column(String(10))
    region = Column(String(10))
    typedisp = Column(String(255))
    quotesourcename = Column(String(255))
    triggerable = Column(Boolean)
    custompricealertconfidence = Column(String(255))
    hasprepostmarketdata = Column(Boolean)
    firsttradedatemilliseconds = Column(Date) 
    postmarketchangepercent = Column(Float)
    postmarketprice = Column(Float)
    postmarketchange = Column(Float)
    regularmarketchange = Column(Float)
    regularmarketdayrange = Column(String(255))
    fullexchangename = Column(String(255))
    averagedailyvolume3month = Column(BigInteger)
    fiftytwoweeklowchange = Column(Float)
    fiftytwoweeklowchangepercent = Column(Float)
    fiftytwoweekrange = Column(String(255))
    fiftytwoweekhighchange = Column(Float)
    fiftytwoweekhighchangepercent = Column(Float)
    fiftytwoweekchangepercent = Column(Float)
    dividenddate = Column(Date) 
    earningstimestamp = Column(DateTime) 
    earningstimestampstart = Column(DateTime) 
    earningstimestampend = Column(DateTime) 
    earningscalltimestampstart = Column(DateTime) 
    earningscalltimestampend = Column(DateTime) 
    isearningsdateestimate = Column(Boolean)
    epstrailingtwelvemonths = Column(Float)
    epsforward = Column(Float)
    epscurrentyear = Column(Float)
    priceepscurrentyear = Column(Float)
    fiftydayaveragechange = Column(Float)
    fiftydayaveragechangepercent = Column(Float)
    twohundreddayaveragechange = Column(Float)
    twohundreddayaveragechangepercent = Column(Float)
    sourceinterval = Column(Integer)
    exchangedatadelayedby = Column(Integer)
    ipoexpecteddate = Column(Date)
    prevname = Column(String(255))
    namechangedate = Column(Date)
    averageanalystrating = Column(String(255))
    cryptotradeable = Column(Boolean)
    postmarkettime = Column(DateTime) 
    regularmarkettime = Column(DateTime) 
    exchange = Column(String(255))
    messageboardid = Column(String(255))
    exchangetimezonename = Column(String(255))
    exchangetimezoneshortname = Column(String(255))
    gmtoffsetmilliseconds = Column(Integer)
    market = Column(String(255))
    esgpopulated = Column(Boolean)
    regularmarketchangepercent = Column(Float)
    regularmarketprice = Column(Float)
    marketstate = Column(String(255))
    shortname = Column(String(255))
    longname = Column(String(255))
    displayname = Column(String(255))
    trailingpegratio = Column(Float)
    irwebsite = Column(String(255), nullable=True)
    fiveyearavgdividendyield = Column(Float, nullable=True)
    lastsplitfactor = Column(String(255))
    lastsplitdate = Column(Date)
    rsratio = Column(Float)

    # CANSLIM Pre-calculated Metrics
    eps_cagr_3year = Column(Float, nullable=True)  # 3-year EPS compound annual growth rate
    price_relative_to_52week_high = Column(Float, nullable=True)  # Current price relative to 52-week high (%)
    expanding_volume = Column(Boolean, nullable=True) # if volume is expanding
    last_revenue_report_date = Column(Date, nullable=True) # Last revenue report date
    last_eps_report_date = Column(Date, nullable=True) # Last EPS report date
    last_roe_report_date = Column(Date, nullable=True) # Last ROE report date
    last_dte_report_date = Column(Date, nullable=True) # Last DTE report date
    last_price_date = Column(Date, nullable=True) # Last Price Date
    relative_strength_percentile_252 = Column(Float, nullable=True) # Relative Strength (Percentile) 1 Year
    relative_strength_percentile_126 = Column(Float, nullable=True) # Relative Strength (Percentile) 6 Months
    relative_strength_percentile_63 = Column(Float, nullable=True) # Relative Strength (Percentile) 3 Months
    revenuegrowth_quarterly_yoy = Column(Float, nullable=True)
    earningsgrowth_quarterly_yoy = Column(Float, nullable=True)
    
    # New Ratios
    netincomegrowth = Column(Float, nullable=True)
    assetturnover = Column(Float, nullable=True)
    inventoryturnoverratio = Column(Float, nullable=True)
    dayspayableoutstanding = Column(Float, nullable=True)
    dayssalesoutstanding = Column(Float, nullable=True)
    dividendpayoutratio = Column(Float, nullable=True)
    interestcoverageratio = Column(Float, nullable=True)
    cashflowtodebtratio = Column(Float, nullable=True)
    cashratio = Column(Float, nullable=True)
    debttoassetsratio = Column(Float, nullable=True)
    debttocash = Column(Float, nullable=True)
    last_other_ratios_report_date = Column(Date, nullable=True)

    # Fundamental score columns
    fundamental_score = Column(Float, nullable=True)
    fundamental_score_percentile = Column(Float, nullable=True)
    fundamental_score_sector_percentile = Column(Float, nullable=True)

    # New trend metrics
    consecutive_quarters_revenue_growth = Column(Integer)
    consecutive_quarters_eps_growth = Column(Integer)
    consecutive_quarters_opmargin_improvement = Column(Integer) # Example for Operating Margin
    revenue_growth_acceleration_qoq_yoy = Column(Float)
    eps_growth_acceleration_qoq_yoy = Column(Float)
    opmargin_improvement_acceleration_qoq_yoy = Column(Float) # Example for Operating Margin
    fcf_growth_qoq = Column(Float)
    fcf_growth_ttm_yoy = Column(Float)
    share_change_rate_qoq = Column(Float) # Negative value indicates reduction
    consecutive_quarters_share_reduction = Column(Integer)

    # Relationships
    financials = relationship("Financials", back_populates="company")
    price_histories = relationship("PriceHistory", back_populates="company")
    officers = relationship("CompanyOfficer", back_populates="company")
    upgrades_downgrades = relationship("UpgradeDowngrade", back_populates="company")
    institutional_holdings = relationship("InstitutionalHolding", back_populates="company")
    insider_transactions = relationship("InsiderTransaction", back_populates="company")
    insider_rosters = relationship("InsiderRoster", back_populates="company")
    earnings_estimates = relationship("AnalystEarningsEstimate", back_populates="company")
    revenue_estimates = relationship("AnalystRevenueEstimate", back_populates="company")
    growth_estimates = relationship("AnalystGrowthEstimate", back_populates="company")
    earnings_history = relationship("AnalystEarningsHistory", back_populates="company")
    eps_trends = relationship("AnalystEpsTrend", back_populates="company")
    eps_revisions = relationship("AnalystEpsRevisions", back_populates="company")

class PriceHistory(Base):
    __tablename__ = "price_history"
    id = Column(Integer, Sequence('price_history_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"), index=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    timeframe = Column(String(10), index=True, nullable=False, server_default='1d') # e.g., '1d', '1h', '15m'
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adjclose = Column(Float)
    volume = Column(BigInteger)
    dividend_amount = Column(Float)
    split_coefficient = Column(Float)

    # Define unique constraint
    __table_args__ = (
        UniqueConstraint('company_id', 'timeframe', 'timestamp', name='uq_company_timeframe_timestamp'),
    )
    # Relationships
    company = relationship("Company", back_populates="price_histories")


class Financials(Base):
    __tablename__ = "financials"
    id = Column(Integer, Sequence('financials_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    report_date = Column(Date, index=True)
    type = Column(String(255))
    index = Column(String(255), index=True)
    value = Column(Float)

    # Define unique constraint
    __table_args__ = (
        UniqueConstraint('company_id', 'report_date', 'type', 'index', name='uq_company_report_type_index'),
    )
    # Relationships
    company = relationship("Company", back_populates="financials")


class CompanyOfficer(Base):
    __tablename__ = "company_officer"

    id = Column(Integer, Sequence('company_officer_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    maxage = Column(Integer)
    name = Column(String(255))
    age = Column(Integer)
    title = Column(String(255))
    yearborn = Column(Integer)
    fiscalyear = Column(Integer)
    totalpay = Column(Integer)
    exercisedvalue = Column(Integer)
    unexercisedvalue = Column(Integer)

    company = relationship("Company", back_populates="officers")

class Exchange(Base):
    __tablename__ = "exchange"

    id = Column(Integer, primary_key=True)
    continent = Column(String(255), nullable=False)
    country = Column(String(255), nullable=False)
    country_code = Column(String(255), nullable=False)
    exchange_code = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    suffix = Column(String(255), nullable=True)
    open_time = Column(String(255), nullable=False)
    close_time = Column(String(255), nullable=False)
    timezone = Column(String(255), nullable=False)


class AnalystEarningsEstimate(Base):
    __tablename__ = "analyst_earnings_estimate"
    id = Column(Integer, Sequence('analyst_earnings_estimate_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    period_label = Column(String(50), index=True) # e.g., "0q", "+1q", "0y", "+1y"
    num_analysts = Column(Integer, nullable=True)
    avg_estimate = Column(Float, nullable=True)
    low_estimate = Column(Float, nullable=True)
    high_estimate = Column(Float, nullable=True)
    year_ago_eps = Column(Float, nullable=True)
    eps_growth_percent = Column(Float, nullable=True) # 'EPS Growth' from yfinance
    revenue_growth_percent = Column(Float, nullable=True) # 'Revenue Growth' from yfinance
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint('company_id', 'period_label', name='uq_analyst_earnings_estimate_key'),)
    company = relationship("Company", back_populates="earnings_estimates")

class AnalystRevenueEstimate(Base):
    __tablename__ = "analyst_revenue_estimate"
    id = Column(Integer, Sequence('analyst_revenue_estimate_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    period_label = Column(String(50), index=True) # e.g., "0q", "+1q", "0y", "+1y"
    num_analysts = Column(Integer, nullable=True)
    avg_estimate = Column(BigInteger, nullable=True) # Revenue is usually a large number
    low_estimate = Column(BigInteger, nullable=True)
    high_estimate = Column(BigInteger, nullable=True)
    year_ago_revenue = Column(BigInteger, nullable=True)
    revenue_growth_percent = Column(Float, nullable=True) # 'Growth' from yfinance
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint('company_id', 'period_label', name='uq_analyst_revenue_estimate_key'),)
    company = relationship("Company", back_populates="revenue_estimates")

class AnalystGrowthEstimate(Base):
    __tablename__ = "analyst_growth_estimate"
    id = Column(Integer, Sequence('analyst_growth_estimate_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    period_label = Column(String(255), index=True) # e.g., "Next 5 Years (per annum)"
    growth_value_text = Column(String(50), nullable=True) # Stores values like "11.90%" or "NaN"
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint('company_id', 'period_label', name='uq_analyst_growth_estimate_key'),)
    company = relationship("Company", back_populates="growth_estimates")

class AnalystEarningsHistory(Base):
    __tablename__ = "analyst_earnings_history"
    id = Column(Integer, Sequence('analyst_earnings_history_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    report_date = Column(Date, index=True)
    eps_estimate = Column(Float, nullable=True)
    eps_actual = Column(Float, nullable=True)
    eps_difference = Column(Float, nullable=True)
    surprise_percent = Column(Float, nullable=True)

    __table_args__ = (UniqueConstraint('company_id', 'report_date', name='uq_analyst_earnings_history_key'),)
    company = relationship("Company", back_populates="earnings_history")

class AnalystEpsTrend(Base):
    __tablename__ = "analyst_eps_trend"
    id = Column(Integer, Sequence('analyst_eps_trend_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    period_label = Column(String(50), index=True) # e.g., "0q", "+1q", "0y", "+1y"
    current_estimate = Column(Float, nullable=True)
    seven_days_ago = Column(Float, nullable=True)
    thirty_days_ago = Column(Float, nullable=True)
    sixty_days_ago = Column(Float, nullable=True)
    ninety_days_ago = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint('company_id', 'period_label', name='uq_analyst_eps_trend_key'),)
    company = relationship("Company", back_populates="eps_trends")

class AnalystEpsRevisions(Base):
    __tablename__ = "analyst_eps_revisions"
    id = Column(Integer, Sequence('analyst_eps_revisions_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    period_label = Column(String(50), index=True) # e.g., "0q", "+1q", "0y", "+1y"
    up_last_7_days = Column(Integer, nullable=True)
    up_last_30_days = Column(Integer, nullable=True)
    down_last_7_days = Column(Integer, nullable=True)
    down_last_30_days = Column(Integer, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (UniqueConstraint('company_id', 'period_label', name='uq_analyst_eps_revisions_key'),)
    company = relationship("Company", back_populates="eps_revisions")

class UpgradeDowngrade(Base):
    __tablename__ = "upgrade_downgrade"
    id = Column(Integer, Sequence('upgrade_downgrade_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    date = Column(Date, index=True)
    firm = Column(String(255))
    to_grade = Column(String(255))
    from_grade = Column(String(255), nullable=True)
    action = Column(String(50))

    __table_args__ = (UniqueConstraint('company_id', 'date', 'firm', 'to_grade', 'action', name='uq_upgrade_downgrade_key'),)
    company = relationship("Company", back_populates="upgrades_downgrades")


class InstitutionalHolding(Base):
    __tablename__ = "institutional_holding"
    id = Column(Integer, Sequence('institutional_holding_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    holder_name = Column(String(255), index=True)
    shares = Column(BigInteger)
    date_reported = Column(Date, index=True)
    percent_out = Column(Float)
    value = Column(BigInteger)
    holder_type = Column(String(50)) # 'institutional' or 'mutualfund'

    __table_args__ = (UniqueConstraint('company_id', 'holder_name', 'date_reported', 'holder_type', name='uq_institutional_holding_key'),)
    company = relationship("Company", back_populates="institutional_holdings")


class InsiderTransaction(Base):
    __tablename__ = "insider_transaction"
    id = Column(Integer, Sequence('insider_transaction_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    insider_name = Column(String(255), index=True)
    shares = Column(BigInteger)
    transaction_type = Column(String(255)) # e.g., "Sale", "Purchase", "Award" - from 'Transaction' column
    transaction_code = Column(String(50), nullable=True) # e.g. 'P-Purchase', 'S-Sale', 'A-Award' - from 'Ownership' column
    start_date = Column(Date, index=True)
    value = Column(BigInteger, nullable=True)

    __table_args__ = (UniqueConstraint('company_id', 'insider_name', 'transaction_type', 'start_date', 'shares', name='uq_insider_transaction_key'),)
    company = relationship("Company", back_populates="insider_transactions")


class InsiderRoster(Base):
    __tablename__ = "insider_roster"
    id = Column(Integer, Sequence('insider_roster_id_seq', start=1, increment=1), primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id"))
    name = Column(String(255), index=True)
    position = Column(String(255))
    most_recent_transaction = Column(String(255), nullable=True)
    most_recent_transaction_date = Column(Date, nullable=True)
    shares_owned_directly = Column(BigInteger, nullable=True)
    shares_owned_indirectly = Column(BigInteger, nullable=True)
    # URL is often part of the 'Position' string, not a separate field from yfinance

    __table_args__ = (UniqueConstraint('company_id', 'name', 'position', name='uq_insider_roster_key'),)
    company = relationship("Company", back_populates="insider_rosters")



""" Script for migrating the date column of PriceHistory to timestamp and timeframe
-- Step 1: Add the new columns
ALTER TABLE price_history 
    ADD COLUMN "timestamp" TIMESTAMP WITHOUT TIME ZONE,
    ADD COLUMN timeframe VARCHAR(10);

-- Step 2: Populate the new columns
UPDATE price_history 
SET timeframe = '1d', 
    "timestamp" = date::timestamp;

-- Step 3: Enforce NOT NULL
ALTER TABLE price_history 
    ALTER COLUMN "timestamp" SET NOT NULL,
    ALTER COLUMN timeframe SET NOT NULL;

-- Step 4: Drop old objects
DROP INDEX IF EXISTS ix_price_history_date;
ALTER TABLE price_history DROP CONSTRAINT IF EXISTS uq_company_id_date;
ALTER TABLE price_history DROP COLUMN date;

-- Step 5: Create new indexes & constraint
CREATE INDEX ix_price_history_company_id ON price_history (company_id);
CREATE INDEX ix_price_history_timeframe ON price_history (timeframe);
CREATE INDEX ix_price_history_timestamp ON price_history ("timestamp");

ALTER TABLE price_history 
    ADD CONSTRAINT uq_company_timeframe_timestamp UNIQUE (company_id, timeframe, "timestamp");
"""