<?php

namespace App\Http\Requests\SurveillanceEvent;

use Illuminate\Foundation\Http\FormRequest;

class UpdateSurveillanceStatusRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'status' => 'required|in:new,acknowledged,dismissed',
        ];
    }
}
