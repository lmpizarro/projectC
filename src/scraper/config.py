urls = {
    "nasdaq100": "https://www.slickcharts.com/nasdaq100",
    "dowjones": "https://www.slickcharts.com/dowjones",
    "sp500": "https://www.slickcharts.com/sp500",
    "sectors": "https://topforeignstocks.com/indices/components-of-the-sp-500-index",
    "cedears": "https://www.rava.com/cotizaciones/cedears",
    "bonos_rava": "https://www.rava.com/perfil",
    "bonistas_com": "https://bonistas.com",
    "treasury_yield_curve": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=",
    "treasury_bill_rates": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=",
}


def url_treasury_by_year(year: int = 2023):
    return f'{urls["treasury_yield_curve"]}{year}'
