from shiny import Inputs, Outputs, Session, App, ui, render, reactive, req
from model_api import APIWrapper
import pandas as pd

api = APIWrapper()

app_ui = ui.page_fluid(
    ui.card(
        ui.row(
            ui.column(
                9,
                ui.div(
                    ui.input_text_area(
                        "text",
                        "Enter Text",
                        height="600px",
                        width="100%",
                        placeholder="Enter text to annotate",
                    ),
                    style='{"color "red"}',
                ),
                ui.layout_column_wrap(
                    ui.input_action_button(
                        "is_electronics", "Electonics", class_="btn btn-primary"
                    ),
                    ui.input_action_button(
                        "not_electronics", "Not Electronics", class_="btn btn-warning"
                    ),
                    ui.input_action_button("skip", "Skip", class_="btn btn-success"),
                    width=1 / 3,
                ),
            ),
            ui.column(
                3,
                ui.value_box(
                    "Model score",
                    ui.output_text("model_score"),
                    fill=False,
                    theme="primary",
                ),
            ),
        )
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    if session.user is None:
        session.user = "local_dev"

    # Populate initial prompt from the database
    update_prompt()

    @reactive.Effect
    @reactive.event(input.skip)
    def skip_prompt():
        update_prompt()

    @reactive.Effect
    @reactive.event(input.is_electronics)
    def mark_electronic():
        annotate_data(input.text(), session.user, True)
        ui.notification_show("Marked electronic", duration=1)
        update_prompt()

    @reactive.Effect
    @reactive.event(input.not_electronics)
    def mark_not_electronic():
        annotate_data(input.text(), "test", False)
        ui.notification_show("Marked not electronic", duration=1)
        update_prompt()

    @render.text
    def model_score():
        req(input.text())
        return round(float(api.score_model(input.text())), 2)


def annotate_data(text, user, annotation):
    df = pd.DataFrame(
        [[text, user, annotation]], columns=["text", "annotator", "annotation"]
    )
    api.upload_data(df)


def update_prompt():
    random_text = api.query_data(
        "SELECT text FROM training_data ORDER BY RANDOM() LIMIT 1"
    )
    random_text_str = random_text["text"][0]
    ui.update_text_area("text", value=random_text_str)


app = App(app_ui, server)
