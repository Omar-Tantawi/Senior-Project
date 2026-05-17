<?php

namespace App\Http\Requests\SurveillanceEvent;

use Illuminate\Foundation\Http\FormRequest;

class SurveillanceSummaryRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'from' => 'required|date',
            'to'   => 'required|date|after_or_equal:from',
        ];
    }
}
