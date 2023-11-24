from abc import ABC, abstractmethod
import numpy as np


class AbstractHittingTimeWithExtentsModel(ABC):

    def __init__(self, length, name):

        self._length = length
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def calculate_ejection_windows(self, q):
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')


class HittingTimeWithExtentsModel(AbstractHittingTimeWithExtentsModel):

    def __init__(self, length, hitting_time_model_class, hitting_time_model_kwargs, name):

        super().__init__(length=length,
                         name=name,
                         )

        htm_kwargs = hitting_time_model_kwargs.copy()
        x_predTo = htm_kwargs.pop('x_predTo')

        self._front_arrival_model = hitting_time_model_class(x_predTo=x_predTo - length / 2,
                                                             **htm_kwargs)
        self._back_arrival_model = hitting_time_model_class(x_predTo=x_predTo + length / 2,
                                                            **htm_kwargs)

    @property
    def front_arrival_model(self):
        return self._front_arrival_model

    @property
    def back_arrival_model(self):
        return self._back_arrival_model

    def calculate_ejection_windows(self, q):
        q_front = (1 - q) / 2
        q_back = (1 + q) / 2

        t_start = self._front_arrival_model.ppf(q_front)
        t_end = self._back_arrival_model.ppf(q_back)
        return t_start, t_end


class HittingTimeWithExtentsSimplifiedModel(AbstractHittingTimeWithExtentsModel):

    def __init__(self, length, hitting_time_model_class, hitting_time_model_kwargs, name):

        super().__init__(length=length,
                         name=name,
                         )

        self._arrival_model = hitting_time_model_class(**hitting_time_model_kwargs)

    def calculate_ejection_windows(self, q):
        q_front = (1 - q) / 2
        q_back = (1 + q) / 2

        t_start = self._arrival_model.ppf(q_front) - self._length/(2 * self._arrival_model._x_L[1])   # TODO: Ist das überhaupt so wie gedacht?
        t_end = self._arrival_model.ppf(q_back) + self._length/(2 * self._arrival_model._x_L[1])
        return t_start, t_end


class AbstractHittingLocationWithExtentsModel(ABC):

    def __init__(self, length, width, name):

        self._length = length
        self._width = width
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def calculate_ejection_windows(self, q):
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')


class HittingLocationWithExtentsModel(AbstractHittingLocationWithExtentsModel):

    def __init__(self,
                 length,
                 width,
                 hitting_time_model_class,
                 hitting_location_model_class,
                 hitting_time_model_kwargs,
                 hitting_location_model_kwargs,
                 name,
                 ):
        super().__init__(length=length,
                         width=width,
                         name=name,
                         )

        htm_kwargs = hitting_time_model_kwargs.copy()
        x_predTo = htm_kwargs.pop('x_predTo')

        self._front_arrival_model = hitting_time_model_class(x_predTo=x_predTo - length / 2,  # TODO: Die ganzen selfs hier entfernen
                                                             **htm_kwargs)
        self._back_arrival_model = hitting_time_model_class(x_predTo=x_predTo + length / 2,
                                                            **htm_kwargs)
        self._front_location_model = hitting_location_model_class(hitting_time_model=self._front_arrival_model,
                                                                  **hitting_location_model_kwargs)  # TODO: Das geht so allgemein nicht
        self._back_location_model = hitting_location_model_class(hitting_time_model=self._back_arrival_model,
                                                                 **hitting_location_model_kwargs)  # TODO: Das geht so allgemein nicht
        self._half_location_model = HalfLocationModel(self._front_location_model, self._back_location_model)

        self._min_y_model = MinYModel(self._front_location_model, self._back_location_model, self._width)

        self._max_y_model = MaxYModel(self._front_location_model, self._back_location_model, self._width)

    @property
    def front_location_model(self):
        return self._front_location_model

    @property
    def back_location_model(self):
        return self._back_location_model

    @property
    def max_y_model(self):
        return self._max_y_model

    @property
    def min_y_model(self):
        return self._min_y_model

    def calculate_ejection_windows(self, q):
        q_low = (1 - q) / 2
        q_up = (1 + q) / 2

        y_start = self._min_y_model.ppf(q_low)
        y_end = self._max_y_model.ppf(q_up)
        return y_start, y_end


class HalfLocationModel(object):

    def __init__(self, front_location_model, back_location_model):
        self._front_location_model = front_location_model
        self._back_location_model = back_location_model

    def pdf(self, y):
        return 1 / 2 * (self._front_location_model.pdf(y) + self._back_location_model.pdf(y))

    def cdf(self, y):
        return 1 / 2 * (self._front_location_model.cdf(y) + self._back_location_model.cdf(y))


class MaxYModel(object):

    def __init__(self, front_location_model, back_location_model, width):
        self._front_location_model = front_location_model
        self._back_location_model = back_location_model
        self._width = width

    def cdf(self, y):
        # by independence assumption
        # return self._front_location_model.cdf(y - self._width / 2) * self._back_location_model.cdf(y - self._width / 2)
        # by no return assumption
        # return self._back_location_model.cdf(y - self._width / 2)
        # return self._front_location_model.cdf(y - self._width / 2)
        # back_location_value = self._back_location_model.cdf(y - self._width / 2)
        # front_location_value = self._front_location_model.cdf(y - self._width / 2)
        return np.min(np.array([self.back_location_cdf(y), self.front_location_cdf(y)]))

    def pdf(self, y):
        # by no return assumption
        # return self._back_location_model.pdf(y - self._width / 2)
        # return self._front_location_model.pdf(y - self._width / 2)
        # back_location_value = self._back_location_model.pdf(y - self._width / 2)
        # front_location_value = self._front_location_model.pdf(y - self._width / 2)
        # # half_model__value = self._half_location_model.pdf(y - self._width / 2)
        # back_location_cdf_value = self._back_location_model.cdf(y - self._width / 2)
        # front_location_cdf_value = self._front_location_model.cdf(y - self._width / 2)
        return np.array([self.back_location_pdf(y), self.front_location_pdf(y)])[
            np.argmin(np.array([self.back_location_cdf(y), self.front_location_cdf(y)]))]

    def back_location_cdf(self, y):
        return self._back_location_model.cdf(y - self._width / 2)

    def front_location_cdf(self, y):
        return self._front_location_model.cdf(y - self._width / 2)

    def back_location_pdf(self, y):
        return self._back_location_model.pdf(y - self._width / 2)

    def front_location_pdf(self, y):
        return self._front_location_model.pdf(y - self._width / 2)

    def ppf(self, q):
        # by no return assumption
        # return self._back_location_model.ppf(q) + self._width / 2
        # return self._front_location_model.ppf(q) + self._width / 2
        back_location_value = self._back_location_model.ppf(q) + self._width / 2
        front_location_value = self._front_location_model.ppf(q) + self._width / 2
        return np.max(np.array([back_location_value, front_location_value]))

    def cdf_values(self, range):
        cdf_values = [self.cdf(ys) for ys in range]
        return cdf_values

    def pdf_values(self, range):
        pdf_values = np.gradient(self.cdf_values(range), range)
        return pdf_values


class MinYModel(object):

    def __init__(self, front_location_model, back_location_model, width):
        self._front_location_model = front_location_model
        self._back_location_model = back_location_model
        self._width = width

    def cdf(self, y):
        # # by independence assumption
        # return 1 - (1 - self._front_location_model.cdf(y + self._width / 2)) * (
        #         1 - self._back_location_model.cdf(y + self._width / 2))
        # by no return assumption
        # return self._back_location_model.cdf(y + self._width / 2)
        # back_location_value = self._back_location_model.cdf(y + self._width / 2)
        # front_location_value = self._front_location_model.cdf(y + self._width / 2)
        return np.max(np.array([self.back_location_cdf(y), self.front_location_cdf(y)]))

    def pdf(self, y):
        # by no return assumption
        # return self._back_location_model.pdf(y + self._width / 2)
        # back_location_value = self._back_location_model.pdf(y + self._width / 2)
        # front_location_value = self._front_location_model.pdf(y + self._width / 2)
        # # half_model_value = self._half_location_model.pdf(y + self._width / 2)
        # back_location_cdf_value = self._back_location_model.cdf(y + self._width / 2)
        # front_location_cdf_value = self._front_location_model.cdf(y + self._width / 2)
        return np.array([self.back_location_pdf(y), self.front_location_pdf(y)])[
            np.argmax(np.array([self.back_location_cdf(y), self.front_location_cdf(y)]))]

    def back_location_cdf(self, y):
        return self._back_location_model.cdf(y + self._width / 2)

    def front_location_cdf(self, y):
        return self._front_location_model.cdf(y + self._width / 2)

    def back_location_pdf(self, y):
        return self._back_location_model.pdf(y + self._width / 2)

    def front_location_pdf(self, y):
        return self._front_location_model.pdf(y + self._width / 2)

    def ppf(self, q):
        # by no return assumption
        # return self._back_location_model.ppf(q) - self._width / 2
        back_location_value = self._back_location_model.ppf(q) - self._width / 2
        front_location_value = self._front_location_model.ppf(q) - self._width / 2
        return np.min(np.array([back_location_value, front_location_value]))

    def cdf_values(self, range):
        cdf_values = [self.cdf(ys) for ys in range]
        return cdf_values

    def pdf_values(self, range):
        pdf_values = np.gradient(self.cdf_values(range), range)
        return pdf_values
